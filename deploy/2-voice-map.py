#!/usr/bin/env python3
"""Print the speaker map from a trained voice config, and the yaml to serve it.

    python3 deploy/2-voice-map.py voices/en_AU-librivox-medium.onnx.json
    python3 deploy/2-voice-map.py ... --raw             # LibriVox narrator names
    python3 deploy/2-voice-map.py ... --write-rates     # create the rate variants
    python3 deploy/2-voice-map.py ... --write           # save voices/voice_to_speaker.yaml

deploy/0-export.sh runs this with --write after every export, so the yaml is
regenerated from the trained config rather than hand-maintained.

Speaker IDs come from the trained config and nowhere else. piper assigns them
during training and they do NOT follow the export roster order -- on this model
Ophelia Darcy came out as speaker 7 while the roster had her at 2. Never infer
an ID from a list; read it from here.

Serving names and speech rates come from voice-names.json beside this script.

Rates need a wrinkle: length_scale lives in the MODEL's config, not per
speaker, so voices that want a different rate need their own .onnx.json. The
.onnx itself is shared via a symlink, so a rate variant costs a few kilobytes
rather than another 63 MB. --write-rates creates them; without it the commands
are printed for you to run.
"""
import argparse
import io
import json
import os
import sys

# This script lives in deploy/, so the repo root is one level up.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES_PATH = os.path.join(HERE, "config", "voice-names.json")


def render_yaml(body):
    """Indent a body under tts-1, which is how the serving layer reads it."""
    return "tts-1:\n" + "".join("  " + line + "\n" for line in body)


def write_yaml(text):
    """Save the serving yaml beside the voice it describes."""
    out = os.path.join(HERE, "voices", "voice_to_speaker.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print()
    print("  wrote %s" % out)


def load_names():
    path = NAMES_PATH
    if not os.path.exists(path):
        return {}, None
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out[k] = (v.get("name", k), v.get("length_scale"))
        else:
            out[k] = (v, None)
    return out, path


def load_hours():
    """Per-speaker hours from prepare/2-export.py, for the comments. Optional."""
    for rel in ("dataset/speakers.json",):
        path = os.path.join(HERE, rel)
        if os.path.exists(path):
            try:
                with io.open(path, encoding="utf-8") as f:
                    d = json.load(f)
                return {s["speaker"]: (s.get("sex", ""), s.get("hours"))
                        for s in d["speakers"]}
            except Exception:
                return {}
    return {}


def variant_stem(base_stem, scale):
    """en_AU-librivox-medium + 1.25 -> en_AU-librivox-medium-ls125"""
    return "%s-ls%s" % (base_stem, ("%.2f" % scale).replace("0.", "").replace(".", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="the trained <voice>.onnx.json")
    ap.add_argument("--raw", action="store_true",
                    help="use LibriVox names and the trained rate for everything")
    ap.add_argument("--write-rates", action="store_true",
                    help="actually create the rate-variant configs and symlinks")
    ap.add_argument("--write", action="store_true",
                    help="also save voices/voice_to_speaker.yaml")
    args = ap.parse_args()

    with io.open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    model = os.path.basename(args.config)[:-5]      # strip .json -> the .onnx name
    stem = model[:-5] if model.endswith(".onnx") else model
    base_ls = cfg.get("inference", {}).get("length_scale", 1.0)

    smap = cfg.get("speaker_id_map") or {}
    names, names_path = ({}, None) if args.raw else load_names()
    hours = load_hours()

    print("  model         %s" % model)
    print("  sample_rate   %s" % cfg.get("audio", {}).get("sample_rate"))
    print("  num_speakers  %s" % cfg.get("num_speakers", 1))
    print("  length_scale  %s  (trained rate; HIGHER IS SLOWER)" % base_ls)
    if names_path:
        print("  names/rates   %s" % os.path.relpath(names_path, HERE))
    print()

    if not smap:
        serving = list(names.values())[0][0] if names else "voice"
        print("  single-speaker voice -- no speaker id needed")
        print()
        print("  %s:" % serving)
        print("    model: voices/%s" % model)
        print("    speaker: # default speaker")
        if args.write:
            write_yaml(render_yaml([
                "# %s -- single speaker" % model,
                "%s:" % serving,
                "  model: voices/%s" % model,
                "  speaker: # default speaker",
            ]))
        return

    rows = []
    for librivox, sid in sorted(smap.items(), key=lambda kv: kv[1]):
        serving, scale = names.get(librivox, (librivox, None))
        sex, hrs = hours.get(librivox, ("", None))
        rows.append((sid, serving, librivox, sex, hrs, scale))

    print("  %-3s %-16s %-26s %-14s %s" % ("id", "serving", "librivox narrator",
                                           "audio", "rate"))
    print("  " + "-" * 74)
    for sid, serving, librivox, sex, hrs, scale in rows:
        h = "%.2f h %s" % (hrs, sex) if hrs else ""
        r = "%.2f (slower)" % scale if scale and scale > base_ls else (
            "%.2f (faster)" % scale if scale else "")
        print("  %-3d %-16s %-26s %-14s %s" % (sid, serving, librivox, h, r))

    unmapped = [r[2] for r in rows if r[1] == r[2]]
    if unmapped and not args.raw:
        print()
        print("  ! not in voice-names.json, keeping LibriVox name: %s"
              % ", ".join(unmapped))

    # --- rate variants -----------------------------------------------------
    scales = sorted({r[5] for r in rows if r[5] and r[5] != base_ls})
    made = {}
    if scales:
        print()
        print("Rate variants -- length_scale is per MODEL, not per speaker, so a")
        print("voice at a different rate needs its own config. The .onnx is shared")
        print("by symlink, so each variant costs kilobytes, not 63 MB.")
        print()
        for sc in scales:
            vstem = variant_stem(stem, sc)
            vonnx = vstem + ".onnx"
            vjson = vonnx + ".json"
            made[sc] = vonnx
            onnx_path = os.path.join(cfg_dir, vonnx)
            json_path = os.path.join(cfg_dir, vjson)
            if args.write_rates:
                if not os.path.exists(onnx_path):
                    try:
                        os.symlink(model, onnx_path)
                    except (OSError, NotImplementedError, AttributeError):
                        import shutil
                        shutil.copy2(os.path.join(cfg_dir, model), onnx_path)
                v = json.loads(json.dumps(cfg))
                v.setdefault("inference", {})["length_scale"] = sc
                with io.open(json_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(v, f, indent=2, ensure_ascii=False)
                print("  wrote %s  (length_scale %.2f)" % (vjson, sc))
            else:
                print("  ln -s %s %s" % (model, vonnx))
                print("  python3 - <<'EOF'")
                print("  import json")
                print("  c = json.load(open('%s'))" % os.path.basename(args.config))
                print("  c.setdefault('inference', {})['length_scale'] = %.2f" % sc)
                print("  json.dump(c, open('%s','w'), indent=2)" % vjson)
                print("  EOF")
        if not args.write_rates:
            print()
            print("  (or re-run with --write-rates to create them)")

    # --- yaml --------------------------------------------------------------
    print()
    yaml_body = ["# %s -- %d Australian voices" % (model, len(rows))]
    for sid, serving, librivox, sex, hrs, scale in rows:
        target = made.get(scale, model) if scale and scale != base_ls else model
        note = librivox if serving != librivox else ""
        if hrs:
            note = ("%s, %.2f h %s" % (note, hrs, sex)).strip(", ").strip()
        if scale and scale != base_ls:
            note += " @ length_scale %.2f" % scale
        yaml_body.append("%s:" % serving)
        yaml_body.append("  model: voices/%s" % target)
        yaml_body.append("  speaker: %d%s"
                         % (sid, ("    # %s" % note) if note else ""))

    print("voice_to_speaker.yaml -- same pattern as the existing libritts_r /")
    print("vctk entries. Paste under tts-1:")
    print()
    for line in ["tts-1:"] + yaml_body:
        print("  " + line)

    if args.write:
        write_yaml(render_yaml(yaml_body))

    print()
    print("(--write saves it to voices/voice_to_speaker.yaml.)")
    print()
    print("Deploy note: put the voice files and this yaml on the serving host.")
    print("If you keep a local mirror of the serving config, re-sync it from")
    print("the host afterwards -- never edit a mirror to record a deploy.")


if __name__ == "__main__":
    main()

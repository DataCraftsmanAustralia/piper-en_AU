---
language:
- en
license: cc-by-4.0
library_name: piper
pipeline_tag: text-to-speech
tags:
- piper
- vits
- text-to-speech
- tts
- australian-english
- en-AU
- onnx
- multi-speaker
- librivox
datasets:
- ablmontazer/australian-english-speech
---

# en_AU-librivox-medium

Ten Australian English voices in one Piper (VITS) model. 22,050 Hz mono, ~70M parameters, 77 MB, runs on CPU. Trained on public domain LibriVox recordings. Piper ships no `en_AU` voice, this fills that gap. The model was trained on a single 96 GB card over 24 hours, it was still improving slowly when training was paused, so there could still be room to improve. The code that built it - dataset prep, training, checkpoint selection by ear - is at [DataCraftsmanAustralia/piper-en_AU](https://github.com/DataCraftsmanAustralia/piper-en_AU).

## Voices

Samples all say *"I reckon we should head down to the beach before it gets too hot."*

<table>
<thead>
<tr><th align="right">id</th><th>voice</th><th>narrator</th><th>gender</th><th>sample</th></tr>
</thead>
<tbody>
<tr><td align="right">0</td><td><strong>clancy</strong></td><td>son_of_the_exiles</td><td>male</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/clancy.wav"></audio></td></tr>
<tr><td align="right">1</td><td><strong>bindi</strong></td><td>jenno</td><td>female</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/bindi.wav"></audio></td></tr>
<tr><td align="right">2</td><td><strong>marlo</strong></td><td>lucy_burgoyne_1950_2014</td><td>female</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/marlo.wav"></audio></td></tr>
<tr><td align="right">3</td><td><strong>kirra</strong></td><td>magdalena</td><td>female</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/kirra.wav"></audio></td></tr>
<tr><td align="right">4</td><td><strong>angus</strong></td><td>algy_pug</td><td>male</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/angus.wav"></audio></td></tr>
<tr><td align="right">5</td><td><strong>banjo</strong></td><td>timothy_ferguson</td><td>male</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/banjo.wav"></audio></td></tr>
<tr><td align="right">6</td><td><strong>flynn</strong></td><td>howard_skyman</td><td>male</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/flynn.wav"></audio></td></tr>
<tr><td align="right">7</td><td><strong>matilda</strong></td><td>ophelia_darcy</td><td>female</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/matilda.wav"></audio></td></tr>
<tr><td align="right">8</td><td><strong>tully</strong></td><td>kirsty_leishman</td><td>female</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/tully.wav"></audio></td></tr>
<tr><td align="right">9</td><td><strong>willow</strong></td><td>jane_bennett</td><td>female</td><td><audio controls preload="none" src="https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/resolve/main/samples/willow.wav"></audio></td></tr>
</tbody>
</table>

## Usage

```sh
pip install piper-tts
echo "Good on ya mate." | python -m piper -m en_AU-librivox-medium.onnx -s 5 -f out.wav
```

```python
import wave
from piper import PiperVoice, SynthesisConfig

voice = PiperVoice.load("en_AU-librivox-medium.onnx")
with wave.open("out.wav", "wb") as f:
    voice.synthesize_wav("Good on ya.", f, SynthesisConfig(speaker_id=5))
```

The `.onnx.json` is required, it holds the speaker map.

**Speech rate.** `length_scale` 1.0 is the trained rate; higher is slower. Set `tully` and `clancy` to 1.25 and `marlo` to 1.08, they rush sometimes, and the samples above are all at 1.0. For serving layers that read the rate from the config instead, `configs/` has ready made variants (`en_AU-librivox-medium-ls125.onnx.json` and `-ls108.onnx.json`); copy the `.onnx` alongside under the matching name. In the code repo, `deploy/2-voice-map.py --write-rates` regenerates them from `config/voice-names.json`.

## Limitations

- **Narrow domain.** All source text is 19th/early 20th century literature. No modern sentences, URLs, emails or timestamps. Numerals, abbreviations and technical terms are the weakest point.
- **Period inflected accent** - formal literary reading, not conversational.
- **Voice quality is inherited** from home recordings: poor mics, volume drift, band limiting. Training more probably won't fix it.
- Transcripts were machine generated (faster-whisper) and uncorrected.
- Not really certain about clancy being Australian, realised after training.

## Licence

**CC BY 4.0.** Commercial use, redistribution and further fine tuning permitted.

Full terms, the ten narrators and provenance: [`ATTRIBUTION.md`](https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium/blob/main/ATTRIBUTION.md).

# Attribution and provenance

`en_AU-librivox-medium` — a ten-speaker Australian English Piper (VITS) voice.

Every second of audio behind this model was recorded by a LibriVox volunteer and
donated to the public domain. This file records who they were, what they read,
and how their recordings became a model.

---

## Provenance chain

| stage | what | licence |
|---|---|---|
| Recordings | LibriVox volunteers, 2008–2023 | **Public domain** |
| Source texts | Public-domain Australian books, via Project Gutenberg | Public domain |
| Dataset | [`ablmontazer/australian-english-speech`](https://huggingface.co/datasets/ablmontazer/australian-english-speech) — 61,662 clips / 110.2 h / 214 readers / 37 books | CC0-1.0 |
| Subset used here | 10 accent-verified speakers, 17,454 clips, **31.01 h** | CC0-1.0 |
| Base checkpoint | `en_GB-jenny_dioco-medium` from [`rhasspy/piper-checkpoints`](https://huggingface.co/datasets/rhasspy/piper-checkpoints), trained on the **Jenny TTS dataset** (Jenny / Dioco) | Commercial use permitted, **attribution to Jenny (Dioco) required** |
| Training | [`OHF-Voice/piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl) v1.8.0, VITS medium, 22,050 Hz | GPL-3.0 (trainer) |

**Shipped model:** epoch 599, 146,999 optimiser steps, ~29.6 h on a 96 GB card
(batch 64, 12 workers, 16-bit mixed precision), fine-tuned from
`en_GB-jenny_dioco-medium` with espeak voice `en-gb-x-rp`. Final metrics
`val_mos` 3.6075 (UTMOS), `val_mel` 0.4626, `loss_d` ~1.92 with no adversarial
collapse across the run.

**Published as:** [`DataCraftsmanAustralia/piper-en_AU-librivox-medium`](https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium)
-- the exported ONNX and its config, the serving map, the `length_scale` variants
and a sample per narrator. `docs/model-card.md` in the code repository is that
model's card.

Selected by listening, not by score. The highest `val_mos` of the project was
3.7609 at epoch 544, but that checkpoint is an outlier against its neighbours
(3.60–3.65) and UTMOS varies ±0.08 between identical runs, so it was not
treated as a meaningful lead. Epoch 599 was preferred by ear.

Worth stating for anyone planning a similar run: **the model was perceptually
converged by roughly epoch 300.** A second 16-hour stretch of training improved
`val_mos` by about 0.14 and produced audio the author could not reliably
distinguish from the 13-hour checkpoint.

---

## About LibriVox

[LibriVox](https://librivox.org) is a volunteer project, founded in 2005 by Hugh
McGuire, that records public-domain books as audiobooks. It is entirely
non-commercial, run by volunteers, and carries no advertising. The texts come
largely from Project Gutenberg — works whose copyright has expired — and anyone
may sign up to read a chapter.

**LibriVox recordings are released into the public domain**, not merely under a
permissive licence. There is no attribution requirement, no non-commercial
clause and no share-alike condition. That is what makes a corpus like this
usable for training a model that can be redistributed and used commercially —
which is unusual among speech datasets of this size.

The attribution in this file is therefore given freely rather than owed. It
seemed the least that could be done.

Three consequences of how LibriVox works shaped this model:

- **Volunteers record at home**, on their own equipment, in their own rooms.
  Audio quality varies considerably between narrators, and some older recordings
  are effectively band-limited.
- **Volunteers are international.** They read whatever interests them, so an
  Australian *book* is no guarantee of an Australian narrator. See
  *Speaker selection* below.
- **Every section carries a spoken intro and outro** mandated by LibriVox. These
  were filtered out of the training data; residual cases are noted under
  *Limitations*.

---

## The narrators

Ten LibriVox narrators, selected from the dataset's 214. Hours are of audio
actually used in training.

| id | speaker | hours | clips | read |
|---:|---|---:|---:|---|
| 0 | `son_of_the_exiles` | 3.39 | 1,669 | *Australian Explorers — Their Labours, Perils, and Achievements*; *Short Poetry Collection 159* |
| 1 | `jenno` | 2.16 | 1,409 | *Australian Explorers*; *Australian Fairy Tales*; *History of Australia and New Zealand from 1606 to 1890* |
| 2 | `lucy_burgoyne_1950_2014` | 1.49 | 934 | *Australian Legendary Tales*; *A Lady's Visit to the Gold Diggings of Australia in 1852-53*; *Selection of Australian Poetry and Prose* |
| 3 | `magdalena` | 6.05 | 3,093 | *Australian Legendary Tales*; *Robert O'Hara Burke and the Australian Exploring Expedition of 1860* |
| 4 | `algy_pug` | 1.80 | 1,185 | *Australian Miscellany*; *Leaves from Australian Forests*; *Short Poetry Collections 092 & 159* |
| 5 | `timothy_ferguson` | 7.97 | 4,002 | *Australian Miscellany*; *Gladstone Colony: An Unwritten Chapter of Australian History* |
| 6 | `howard_skyman` | 0.58 | 492 | *Old Broadbrim into the Heart of Australia* |
| 7 | `ophelia_darcy` | 3.15 | 1,900 | *Seven Little Australians* (Ethel Turner) |
| 8 | `kirsty_leishman` | 3.57 | 2,188 | *Two Sides To Every Question: From A South Australian Standpoint* |
| 9 | `jane_bennett` | 0.86 | 582 | *History of Australia and New Zealand from 1606 to 1890* |

Fifteen distinct works. The names above are the narrators' own LibriVox display
names. If a serving layer renames these voices, please keep the original name
alongside — it is the only link back to the person who made the recording.

The remaining 204 narrators in the source dataset are listed in its `CREDITS.md`.
Their recordings are not in this model, but the dataset would not exist without
them.

---

## Speaker selection

**Accent was verified by ear, not by documentation.** The source dataset is the
Australian *titles* in the LibriVox catalogue, and its own card is explicit that
narrator accent is unverified. Candidate narrators were auditioned in two rounds
and ten were judged to have Australian accents by a native listener.

This is a subjective judgement by one person, and it is worth saying plainly
that it could be wrong in either direction. Notably the largest narrator in the
corpus — 11.79 h, more audio than any speaker in this model — was rejected at
this stage.

Exactly one narrator carries documentary confirmation: Timothy Ferguson's own
spoken outro identifies him as recording from the Gold Coast, Australia.

Narrators were also weighed on transcript quality. The dataset's transcripts are
machine-generated and punctuation is inconsistent between narrators — from 0% to
82% of a narrator's clips ending in terminal punctuation. Since espeak-ng drives
prosody from punctuation, narrators with stripped transcripts contribute flatter
intonation. Two otherwise large narrators were excluded partly on this basis.

---

## Licensing

The audio, the texts and the dataset are all free of restriction: public domain
recordings of public domain books, redistributed under CC0-1.0. Nothing on the
LibriVox side constrains use, redistribution or commercial application.

### The base checkpoint — attribution required

This model was fine-tuned from `en_GB-jenny_dioco-medium`, whose weights derive
from the **Jenny TTS dataset**, recorded by **Jenny** and published by
**Dioco**. Its licence permits commercial use and requires attribution:

> Attribution is required in software/websites/projects/interfaces (including
> voice interfaces) that generate audio in response to user action using this
> dataset. Atribution means: the voice must be referred to as "Jenny", and
> where at all practical, "Jenny (Dioco)". Attribution is not required when
> distributing the generated clips (although welcome). Commercial use is
> permitted. Don't do unfair things like claim the dataset is your own. No
> further restrictions apply.

**So: commercial use is permitted, and attribution to Jenny (Dioco) is
required** wherever this model is served in software or an interface. That
requirement is met by this file, and should be carried into the model card and
into any application or voice interface built on the model.

One point of interpretation, stated openly. The clause *"the voice must be
referred to as Jenny"* is written for a project using **Jenny's voice**. This
model does not produce Jenny's voice: it was fine-tuned across 299 epochs on
31.01 hours from ten other speakers, each with their own learned speaker
embedding, and its outputs are those narrators, not her. Calling these voices
"Jenny" would misattribute them to someone who did not record them.

The reading taken here is that the requirement attaches to the **lineage**
rather than the output: the Jenny (Dioco) dataset is credited prominently as the
origin of the base checkpoint, and the voices are named for the LibriVox
narrators who actually produced them. That is offered as good-faith compliance
with a licence that did not anticipate this case, not as a legal conclusion.

Jenny takes commissions for further recordings: **dioco@dioco.io**.

*(Incidentally, Jenny is Irish, and the checkpoint is labelled `en_GB` and
phonemised with `en-gb-x-rp`. An Irish-voiced, RP-phonemised base fine-tuned
toward Australian is an odd lineage on paper. In practice all espeak English
voices share one phoneme inventory, so the transfer is clean — and it evidently
worked.)*

---

## Limitations

- **Transcripts are machine-generated** (faster-whisper large-v3-turbo) and were
  not manually corrected. Punctuation and capitalisation are inconsistent.
- **The data is lopsided** — 7.97 h for the largest speaker against 0.58 h for
  the smallest. This mattered **less than expected**; see below.
- **The voices are not equally good.** All ten assessed by ear on the shipped
  model, 2026-09-20:

  | voice | narrator | hours | punct | pacing | voice quality |
  |---|---|---:|---:|---|---|
  | `banjo` | timothy_ferguson | 7.97 | 21% | good | **great — best overall** |
  | `kirra` | magdalena | 6.05 | 73% | good | audibly poor microphone |
  | `tully` | kirsty_leishman | 3.57 | 36% | rushes | below average, but a great voice underneath |
  | `clancy` | son_of_the_exiles | 3.39 | 37% | rushes | — |
  | `matilda` | ophelia_darcy | 3.15 | 80% | mixed — excellent on some words | decent |
  | `bindi` | jenno | 2.16 | 55% | good | good |
  | `angus` | algy_pug | 1.80 | 27% | good | variable, volume drifts. **Most expressive — real passion** |
  | `marlo` | lucy_burgoyne | 1.49 | 66% | sometimes rushes | — |
  | `willow` | jane_bennett | 0.86 | 45% | **perfect** | the only flaw |
  | `flynn` | howard_skyman | 0.58 | 78% | good | flat, unexpressive |

  **Voice quality is the dominant limitation, and all of it is inherited.**
  Microphone quality, volume drift and flatness come from the source
  recordings — LibriVox volunteers record at home, across sessions, on their
  own gear. No amount of training fixes them.

  **Pacing is the narrator's own reading speed, reproduced faithfully.** It
  tracks neither training volume nor transcript punctuation: `banjo` has the
  worst punctuation rate in the corpus and good pacing, `willow` has 51 minutes
  of audio and perfect pacing, `tully` has 3.57 h and rushes. Because it is
  fidelity rather than failure, it is corrected at inference with piper's
  `length_scale` rather than by retraining — `tully`, `clancy` and `marlo` ship
  with raised values.

  Filtering the training set to punctuated clips only was measured and
  rejected: it would have cut `banjo` by 87% and `angus` by 78% to help
  `clancy`, taking the corpus from 31.01 h to 12.24 h.

- **Speaker imbalance did not degrade the small voices.** The expectation going
  in was that under-represented speakers in multi-speaker VITS drift toward
  generic output or toward the dominant voices. That did not happen here.
  `willow` (0.86 h) has the best pacing in the model and `flynn` (0.58 h) is
  competent; both are limited by their source audio, not by their share of the
  training set. The 31 hours of shared backbone appears to have carried them.
  Recorded because it is evidence against a common assumption, on a single
  model with subjective assessment — not a general finding.
- **The domain is narrow.** Every source text is 19th- and early 20th-century
  Australian literature. The model has never seen a modern sentence, a URL, an
  email address or a timestamp. Timing and emphasis on contemporary text —
  numerals, abbreviations, technical terms — are the model's weakest point.
- **Recording conditions vary** with each narrator's home setup, and some older
  source recordings are band-limited.
- **The accent is period-inflected.** These narrators are reading Australian
  literature aloud in a formal register. The result is Australian, but not
  conversational Australian.
- A small number of LibriVox intro/outro announcements survived the source
  dataset's own filtering and were removed here — 117 clips, 0.67% of the
  subset. Any that remain would appear as unusually formal, repeated phrasing.

---

## A note on the voices

These recordings are public domain, and nothing here requires permission. It is
still worth stating that the narrators donated their readings years ago, to make
books freely available, at a time when training a speech model on them was not a
consideration anyone weighed. That is not a legal problem; the public domain
dedication is unconditional and genuinely meant. It is a reason to credit them
by name, to say what they read, and not to represent these voices as belonging
to anyone other than the people who recorded them.

---

## Acknowledgements

- The **214 LibriVox narrators** in the source dataset, and the ten above in
  particular.
- **LibriVox** and **Project Gutenberg**, for making any of this possible.
- **[`ablmontazer`](https://huggingface.co/datasets/ablmontazer/australian-english-speech)**,
  for assembling and releasing the corpus under CC0.
- **Jenny**, and **Dioco**, for the Jenny TTS dataset behind the
  `en_GB-jenny_dioco-medium` base checkpoint this model was fine-tuned from.
  Commissions: dioco@dioco.io
- **[OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)** and the
  Piper project, and **[rhasspy](https://huggingface.co/rhasspy)** for the
  pretrained checkpoints.
- **DeepFilterNet 3**, **Silero VAD** and **faster-whisper**, used by the source
  dataset for denoising, segmentation and transcription.
- **UTMOS / SpeechMOS** ([tarepan](https://github.com/tarepan/SpeechMOS)), used
  to score checkpoints during training.

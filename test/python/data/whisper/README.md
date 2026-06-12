<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Whisper test audio samples

Small public-domain WAV clips used by the Whisper-large-v3 end-to-end tests
under `test/python/`. **The audio is NOT committed to git** — it is
auto-downloaded at test setup and cached locally in this directory (the WAVs and
`librispeech/` are gitignored). Only this README is tracked, to document
provenance and license. If a download is unreachable (offline / expired URL) the
dependent tests `pytest.skip` rather than error. This file documents where the
data comes from; keep new additions small and CC0 / public domain.

| File | Length | Sample rate | Source | License |
|---|---|---|---|---|
| `jfk.wav` | ~11 s | 16 kHz mono | [ggml-org/whisper.cpp `samples/jfk.wav`](https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav) — JFK inaugural address excerpt. The openai/whisper repo only ships a FLAC version (`tests/jfk.flac`); the whisper.cpp port mirrors the same clip as a 16-bit PCM WAV which is what we use here. Auto-fetched by `conftest.py::setup_jfk_sample`. | Public domain |

## LibriSpeech clips (`librispeech/`)

`librispeech/sample_0.wav` … `sample_4.wav` plus `long_30s.wav` broaden the
Whisper end-to-end coverage across speakers / sentences / lengths (see
`test_whisper.py::test_librispeech_*` and `test_long_30s_gpu_vs_cpu`). These WAVs
and the ground-truth transcript file `librispeech/references.json`
(`{filename: UPPERCASE text}`) are all **generated/written** by
`conftest.py::setup_librispeech_samples` at test setup — none are committed.

- **Source:** the [`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
  `clean/validation` split (the first 5 clips), fetched over the HF datasets-server
  HTTP JSON API (no `datasets` library). These are LibriSpeech clips — read speech
  from public-domain LibriVox audiobooks.
- **License:** **CC BY 4.0** (LibriSpeech). Attribution: *"LibriSpeech: an ASR
  corpus based on public-domain audiobooks", V. Panayotov, G. Chen, D. Povey,
  S. Khudanpur, ICASSP 2015.*
- **Transcoding:** signed FLAC URLs → 16 kHz mono 16-bit PCM WAV via `soundfile`
  (LibriSpeech is already 16 kHz, so no resampling). Reproduced idempotently by
  `conftest.py::setup_librispeech_samples`.
- **`long_30s.wav`:** built by concatenating the first few dummy clips that fit
  under ~29 s (with 0.3 s silence between) — ~24 s total. Its reference is the
  concatenation, in order, of those clips' transcripts. Staying under Whisper's
  30 s window exercises a deep decode (a few hundred tokens toward the 448-slot
  KV cap) without triggering whisper's chunking. Both audio and reference are
  rebuilt by the same helper.

If the datasets-server fetch fails (network / expired signed URL) and the WAVs
aren't already cached, the dependent tests `pytest.skip` rather than error.

## Adding a new sample

1. Confirm the source license is CC0 / public domain (or your own original work).
2. Prefer 16 kHz mono WAV — that matches what `WhisperFeatureExtractor` expects
   and avoids dragging `librosa` into the smoke-test path.
3. Keep the clip small (<1 MB ideally); long clips belong on Hugging Face, not
   in this repo.
4. Add a row to the table above with the source URL and license.

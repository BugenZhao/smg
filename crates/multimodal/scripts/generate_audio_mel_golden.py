#!/usr/bin/env python3
"""Generate a HuggingFace reference log-mel golden for Qwen3 audio preprocessing.

Reference extractor
-------------------
Qwen3-ASR and Qwen3-Omni consume audio through ``WhisperFeatureExtractor``
(``Qwen3OmniMoeProcessor.feature_extractor_class == "WhisperFeatureExtractor"``).
This script constructs that extractor locally with the parameters Qwen3's
``preprocessor_config.json`` sets (128 mel bins, 16 kHz, ``n_fft=400``,
``hop_length=160``), so it needs neither model weights nor a network download.

The extractor is run on a deterministic, seeded synthetic waveform so the golden
is reproducible without any external audio file. Both the input PCM samples and
the reference mel tensor are dumped to a checked-in JSON fixture that the Rust
integration test loads with ``include_str!`` as an external correctness oracle.

Parity note
-----------
SMG's frontend mirrors ``WhisperFeatureExtractor._torch_extract_fbank_features``
applied to the raw (un-padded) waveform: a centered STFT (reflect pad
``n_fft // 2``) yielding ``floor(len / hop) + 1`` frames, with the trailing frame
dropped (``stft[..., :-1]``), Slaney mel filters, and the Whisper log
normalization ``clamp(log10(x), peak - 8); (x + 4) / 4``. We therefore call the
extractor's internal fbank routine directly instead of its public ``__call__``,
which would otherwise pad/truncate every clip to the fixed 30 s window.

Usage::

    python3 crates/multimodal/scripts/generate_audio_mel_golden.py \
        > crates/multimodal/tests/fixtures/golden/audio_mel_reference.json
"""

import json

import numpy as np
import transformers
from transformers import WhisperFeatureExtractor

# Matches Qwen3AudioParams::default() / Qwen3 preprocessor_config.json.
SAMPLE_RATE = 16000
FEATURE_SIZE = 128
N_FFT = 400
HOP_LENGTH = 160

# Deterministic synthetic waveform: a seeded mix of sinusoids plus a little
# reproducible noise, so the golden is stable across machines.
DURATION_SECONDS = 0.75
SEED = 1905  # PR that introduced Qwen3 audio support.


def make_waveform() -> np.ndarray:
    """Build a deterministic mono f32 waveform at ``SAMPLE_RATE``."""
    num_samples = int(round(DURATION_SECONDS * SAMPLE_RATE))
    t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE
    signal = (
        0.6 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 3300.0 * t)
    )
    rng = np.random.default_rng(SEED)
    signal += 0.01 * rng.standard_normal(num_samples)
    # Normalize to a stable peak so the log-mel floor is well defined, then cast
    # to f32 -- the exact dtype the Rust side receives from decode.
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak * 0.95
    return signal.astype(np.float32)


def reference_log_mel(extractor: WhisperFeatureExtractor, waveform: np.ndarray) -> np.ndarray:
    """Reference log-mel for the raw waveform (no fixed 30 s pad/truncate).

    ``_torch_extract_fbank_features`` performs the centered STFT, power spectrum,
    Slaney mel projection, and Whisper log normalization, and already drops the
    trailing STFT frame -- exactly the pipeline SMG reproduces per clip.
    """
    features = extractor._torch_extract_fbank_features(waveform.astype(np.float32))
    return np.asarray(features, dtype=np.float32)


def main() -> None:
    extractor = WhisperFeatureExtractor(
        feature_size=FEATURE_SIZE,
        sampling_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        chunk_length=30,
        padding_value=0.0,
        # dither defaults to 0.0; keep it explicit so the golden is deterministic.
        dither=0.0,
    )

    waveform = make_waveform()
    mel = reference_log_mel(extractor, waveform)
    assert mel.ndim == 2 and mel.shape[0] == FEATURE_SIZE, mel.shape
    # Sanity: HF drops the trailing frame, matching SMG's floor(len / hop).
    assert mel.shape[1] == waveform.shape[0] // HOP_LENGTH, (
        mel.shape,
        waveform.shape[0] // HOP_LENGTH,
    )

    document = {
        "generator": "generate_audio_mel_golden.py",
        "reference": "transformers.WhisperFeatureExtractor",
        "transformers": transformers.__version__,
        "sample_rate": SAMPLE_RATE,
        "n_mels": FEATURE_SIZE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "pcm": [float(sample) for sample in waveform.tolist()],
        "mel_shape": list(mel.shape),
        # Row-major (mel, frame), matching the Rust Array2 layout.
        "mel": [float(value) for value in mel.reshape(-1).tolist()],
        "mel_sum": float(np.sum(mel, dtype=np.float64)),
    }
    print(json.dumps(document))


if __name__ == "__main__":
    main()

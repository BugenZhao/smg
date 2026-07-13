//! Full-tensor parity check for Qwen3 audio log-mel preprocessing.
//!
//! The golden fixture is produced by `scripts/generate_audio_mel_golden.py`
//! from HuggingFace `transformers.WhisperFeatureExtractor` -- the feature
//! extractor Qwen3-ASR / Qwen3-Omni use. It captures both the input PCM and the
//! reference mel tensor so the pure-Rust frontend can be diffed against it here.
//!
//! The bar is a tolerance on the full tensor, not bitwise equality: SMG uses
//! pure-Rust `rustfft`, which cannot be bit-identical to torch/numpy FFT.
#![allow(clippy::expect_used, clippy::panic)]
#![expect(clippy::print_stdout, reason = "integration tests: diagnostic output")]

use llm_multimodal::audio::{DecodedAudio, Qwen3AudioProcessor};
use serde::Deserialize;

/// Max-abs-diff tolerance vs. the transformers golden. The observed diff on the
/// checked-in fixture is ~1e-5 (rustfft vs. torch STFT); 1e-3 leaves ample
/// headroom while still catching any real algorithmic regression.
const MAX_ABS_DIFF_TOL: f32 = 1e-3;

#[derive(Deserialize)]
struct AudioMelGolden {
    generator: String,
    reference: String,
    transformers: String,
    sample_rate: usize,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
    pcm: Vec<f32>,
    mel_shape: Vec<usize>,
    mel: Vec<f32>,
    mel_sum: f64,
}

#[test]
fn qwen3_audio_log_mel_matches_transformers_golden() {
    let golden: AudioMelGolden =
        serde_json::from_str(include_str!("fixtures/golden/audio_mel_reference.json"))
            .expect("invalid checked-in audio mel golden fixture");

    assert_eq!(golden.generator, "generate_audio_mel_golden.py");
    assert_eq!(golden.reference, "transformers.WhisperFeatureExtractor");
    assert!(!golden.transformers.is_empty());
    assert_eq!(golden.mel_shape.len(), 2, "mel golden must be 2-D");
    assert_eq!(
        golden.mel.len(),
        golden.mel_shape[0] * golden.mel_shape[1],
        "mel value count must match declared shape"
    );
    assert!(!golden.pcm.is_empty(), "golden PCM must not be empty");

    // Golden fixture is generated with Qwen3AudioParams::default() parameters.
    let processor = Qwen3AudioProcessor::new();
    let params = processor.params();
    assert_eq!(
        params.sample_rate, golden.sample_rate,
        "sample_rate mismatch"
    );
    assert_eq!(params.n_mels, golden.n_mels, "n_mels mismatch");
    assert_eq!(params.n_fft, golden.n_fft, "n_fft mismatch");
    assert_eq!(params.hop_length, golden.hop_length, "hop_length mismatch");

    let decoded = DecodedAudio {
        samples: golden.pcm.clone(),
        sample_rate: golden.sample_rate,
    };
    let features = processor
        .preprocess_decoded(decoded)
        .expect("Qwen3 audio log-mel preprocessing failed");

    assert_eq!(
        features.shape(),
        golden.mel_shape.as_slice(),
        "log-mel shape differs from transformers golden"
    );

    let values = features
        .as_slice_memory_order()
        .expect("Qwen3 log-mel output must be contiguous");
    assert_eq!(values.len(), golden.mel.len());

    let mut max_abs_diff = 0.0_f32;
    let mut max_at = (0usize, 0usize);
    let frames = golden.mel_shape[1];
    for (index, (&actual, &expected)) in values.iter().zip(&golden.mel).enumerate() {
        let diff = (actual - expected).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_at = (index / frames, index % frames);
        }
    }

    let actual_sum: f64 = values.iter().map(|&value| f64::from(value)).sum();
    let mean_abs_diff: f64 = values
        .iter()
        .zip(&golden.mel)
        .map(|(&actual, &expected)| f64::from((actual - expected).abs()))
        .sum::<f64>()
        / values.len() as f64;

    // Surfaced on every run so the real headroom is visible in CI logs.
    println!(
        "audio mel parity: max_abs_diff={max_abs_diff:.3e} at (mel={}, frame={}), \
         mean_abs_diff={mean_abs_diff:.3e}, tol={MAX_ABS_DIFF_TOL:.0e}, \
         rust_sum={actual_sum:.6}, golden_sum={:.6}",
        max_at.0, max_at.1, golden.mel_sum
    );

    assert!(
        max_abs_diff < MAX_ABS_DIFF_TOL,
        "log-mel max-abs-diff {max_abs_diff:.3e} exceeds tolerance {MAX_ABS_DIFF_TOL:.0e} \
         at (mel={}, frame={})",
        max_at.0,
        max_at.1
    );

    // Sum sanity: catches gross scale/normalization drift the per-cell max diff
    // could miss if it were ever loosened.
    assert!(
        (actual_sum - golden.mel_sum).abs() < 0.05,
        "log-mel sum {actual_sum:.6} differs from golden {:.6}",
        golden.mel_sum
    );
}

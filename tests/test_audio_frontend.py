"""
Tests for the audio_frontend module.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchaudio_aligner.audio_frontend import (
    AudioFrontend,
    AudioSegment,
    SegmentationResult,
    segment_audio,
)


# Test audio file path
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
TEST_AUDIO = EXAMPLES_DIR / "4780182.mp3"


def test_audio_frontend_load():
    """Test loading audio file."""
    print("\n=== Test: AudioFrontend.load() ===")

    frontend = AudioFrontend(target_sample_rate=16000)
    waveform, sample_rate = frontend.load(TEST_AUDIO)

    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")

    assert waveform.dim() == 2, "Waveform should be 2D (channels, samples)"
    assert sample_rate > 0, "Sample rate should be positive"
    print("PASSED")


def test_audio_frontend_resample():
    """Test resampling audio."""
    print("\n=== Test: AudioFrontend.resample() ===")

    frontend = AudioFrontend(target_sample_rate=16000)
    waveform, orig_sr = frontend.load(TEST_AUDIO)

    print(f"Original sample rate: {orig_sr}")
    print(f"Original samples: {waveform.shape[1]}")

    resampled = frontend.resample(waveform, orig_sr, 16000)

    expected_samples = int(waveform.shape[1] * 16000 / orig_sr)
    print(f"Resampled samples: {resampled.shape[1]}")
    print(f"Expected samples (approx): {expected_samples}")

    # Allow some tolerance due to resampling
    assert abs(resampled.shape[1] - expected_samples) < 100, "Resampled length mismatch"
    print("PASSED")


def test_audio_frontend_to_mono():
    """Test converting to mono."""
    print("\n=== Test: AudioFrontend.to_mono() ===")

    frontend = AudioFrontend(target_sample_rate=16000)
    waveform, _ = frontend.load(TEST_AUDIO)

    print(f"Original channels: {waveform.shape[0]}")

    mono = frontend.to_mono(waveform)

    print(f"Mono channels: {mono.shape[0]}")
    assert mono.shape[0] == 1, "Should have 1 channel after mono conversion"
    print("PASSED")


def test_audio_frontend_segment():
    """Test segmentation."""
    print("\n=== Test: AudioFrontend.segment() ===")

    frontend = AudioFrontend(target_sample_rate=16000, mono=True)
    waveform, orig_sr = frontend.load(TEST_AUDIO)
    waveform = frontend.resample(waveform, orig_sr)
    waveform = frontend.to_mono(waveform)

    result = frontend.segment(
        waveform,
        sample_rate=16000,
        segment_size=15.0,
        overlap=2.0,
        min_segment_size=0.2,
    )

    print(f"Original duration: {result.original_duration_seconds:.2f} seconds")
    print(f"Number of segments: {result.num_segments}")
    print(f"Segment size (samples): {result.segment_size_samples}")
    print(f"Overlap (samples): {result.overlap_samples}")

    # Check segments
    for i, seg in enumerate(result.segments[:3]):
        print(f"  Segment {i}: offset={seg.offset_seconds:.2f}s, duration={seg.duration_seconds:.2f}s")
    if result.num_segments > 3:
        print(f"  ... ({result.num_segments - 3} more segments)")

    # Last segment
    last_seg = result.segments[-1]
    print(f"  Last segment: offset={last_seg.offset_seconds:.2f}s, duration={last_seg.duration_seconds:.2f}s")

    assert result.num_segments > 0, "Should have at least one segment"
    assert all(seg.sample_rate == 16000 for seg in result.segments), "All segments should have correct sample rate"
    print("PASSED")


def test_audio_frontend_process():
    """Test full processing pipeline."""
    print("\n=== Test: AudioFrontend.process() ===")

    frontend = AudioFrontend(
        target_sample_rate=16000,
        mono=True,
        normalize=False,
    )

    result = frontend.process(
        TEST_AUDIO,
        segment_size=15.0,
        overlap=2.0,
    )

    print(f"Original duration: {result.original_duration_seconds:.2f} seconds")
    print(f"Number of segments: {result.num_segments}")

    assert isinstance(result, SegmentationResult)
    assert result.num_segments > 0
    print("PASSED")


def test_segmentation_result_batching():
    """Test batching functionality."""
    print("\n=== Test: SegmentationResult.get_waveforms_batched() ===")

    frontend = AudioFrontend(target_sample_rate=16000, mono=True)
    result = frontend.process(TEST_AUDIO, segment_size=15.0, overlap=2.0)

    waveforms, lengths = result.get_waveforms_batched()

    print(f"Batched waveforms shape: {waveforms.shape}")
    print(f"Lengths shape: {lengths.shape}")
    print(f"First few lengths: {lengths[:5].tolist()}")

    assert waveforms.shape[0] == result.num_segments, "Batch size should match num_segments"
    assert lengths.shape[0] == result.num_segments, "Lengths should match num_segments"
    assert waveforms.dim() == 2, "Batched waveforms should be 2D for mono"
    print("PASSED")


def test_segmentation_result_frame_offsets():
    """Test frame offset calculation."""
    print("\n=== Test: SegmentationResult.get_offsets_in_frames() ===")

    frontend = AudioFrontend(target_sample_rate=16000, mono=True)
    result = frontend.process(TEST_AUDIO, segment_size=15.0, overlap=2.0)

    # MMS model has 20ms frame duration
    frame_duration = 0.02
    offsets = result.get_offsets_in_frames(frame_duration)

    print(f"Frame offsets shape: {offsets.shape}")
    print(f"First few offsets (frames): {offsets[:5].tolist()}")

    # Check that offsets are monotonically increasing
    for i in range(1, len(offsets)):
        assert offsets[i] > offsets[i-1], "Offsets should be monotonically increasing"

    print("PASSED")


def test_convenience_function():
    """Test the convenience function."""
    print("\n=== Test: segment_audio() ===")

    result = segment_audio(
        TEST_AUDIO,
        target_sample_rate=16000,
        segment_size=15.0,
        overlap=2.0,
    )

    print(f"Duration: {result.original_duration_seconds:.2f}s")
    print(f"Segments: {result.num_segments}")

    assert isinstance(result, SegmentationResult)
    print("PASSED")


def test_normalization():
    """Test audio normalization."""
    print("\n=== Test: AudioFrontend with normalization ===")

    frontend_no_norm = AudioFrontend(target_sample_rate=16000, mono=True, normalize=False)
    frontend_norm = AudioFrontend(target_sample_rate=16000, mono=True, normalize=True, normalize_db=-3.0)

    waveform, sr = frontend_no_norm.load(TEST_AUDIO)
    waveform = frontend_no_norm.resample(waveform, sr)
    waveform = frontend_no_norm.to_mono(waveform)

    original_peak = waveform.abs().max().item()
    print(f"Original peak: {original_peak:.4f}")

    normalized = frontend_norm.apply_normalization(waveform.clone())
    normalized_peak = normalized.abs().max().item()
    print(f"Normalized peak: {normalized_peak:.4f}")

    expected_peak = 10 ** (-3.0 / 20)  # -3 dB
    print(f"Expected peak (-3dB): {expected_peak:.4f}")

    assert abs(normalized_peak - expected_peak) < 0.01, "Normalized peak should be close to target"
    print("PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Audio Frontend Tests")
    print(f"Test audio: {TEST_AUDIO}")
    print("=" * 60)

    if not TEST_AUDIO.exists():
        print(f"ERROR: Test audio file not found: {TEST_AUDIO}")
        return False

    tests = [
        test_audio_frontend_load,
        test_audio_frontend_resample,
        test_audio_frontend_to_mono,
        test_audio_frontend_segment,
        test_audio_frontend_process,
        test_segmentation_result_batching,
        test_segmentation_result_frame_offsets,
        test_convenience_function,
        test_normalization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

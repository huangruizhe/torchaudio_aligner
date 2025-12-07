"""
Audio Segmentation Module

Divide-and-conquer segmentation for long-form audio alignment.
"""

from dataclasses import dataclass
from typing import List, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata."""
    waveform: torch.Tensor  # Shape: (num_samples,) or (channels, num_samples)
    sample_rate: int
    offset_samples: int  # Offset in samples from the start of the original audio
    length_samples: int  # Actual length of this segment
    segment_index: int  # Index of this segment in the sequence

    @property
    def offset_seconds(self) -> float:
        """Offset in seconds from the start of the original audio."""
        return self.offset_samples / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        """Duration of this segment in seconds."""
        return self.length_samples / self.sample_rate


@dataclass
class SegmentationResult:
    """Result of audio segmentation containing all segments and metadata."""
    segments: List[AudioSegment]
    original_duration_samples: int
    original_duration_seconds: float
    sample_rate: int
    segment_size_samples: int
    overlap_samples: int
    num_segments: int

    def get_waveforms_batched(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack all segment waveforms into a batch tensor.

        Returns:
            waveforms: Tensor of shape (num_segments, max_segment_length)
            lengths: Tensor of shape (num_segments,) with actual lengths
        """
        max_len = max(seg.waveform.shape[-1] for seg in self.segments)
        batch_size = len(self.segments)

        # Determine number of channels
        if self.segments[0].waveform.dim() == 1:
            waveforms = torch.zeros(batch_size, max_len)
        else:
            num_channels = self.segments[0].waveform.shape[0]
            waveforms = torch.zeros(batch_size, num_channels, max_len)

        lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, seg in enumerate(self.segments):
            length = seg.waveform.shape[-1]
            if seg.waveform.dim() == 1:
                waveforms[i, :length] = seg.waveform
            else:
                waveforms[i, :, :length] = seg.waveform
            lengths[i] = length

        return waveforms, lengths

    def get_offsets_in_frames(self, frame_duration_seconds: float) -> torch.Tensor:
        """
        Get segment offsets in frames (for acoustic model output).

        Args:
            frame_duration_seconds: Duration of each frame in seconds (e.g., 0.02 for 20ms)

        Returns:
            Tensor of shape (num_segments,) with frame offsets
        """
        offsets = torch.tensor([seg.offset_samples for seg in self.segments])
        return (offsets / self.sample_rate / frame_duration_seconds).long()


def segment_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_size: float = 15.0,
    overlap: float = 2.0,
    min_segment_size: float = 0.2,
    extra_samples: int = 128,
) -> SegmentationResult:
    """
    Segment audio into overlapping chunks.

    This implements the divide-and-conquer approach: long audio is split
    into overlapping segments that can be processed independently.

    Args:
        waveform: Input waveform tensor of shape (channels, num_samples) or (num_samples,)
        sample_rate: Sample rate of the waveform
        segment_size: Size of each segment in seconds (default 15.0s)
        overlap: Overlap between consecutive segments in seconds (default 2.0s)
        min_segment_size: Minimum size for the last segment in seconds
        extra_samples: Extra samples for edge effects in convolution models

    Returns:
        SegmentationResult containing all segments and metadata
    """
    # Handle different input shapes
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension

    num_channels, total_samples = waveform.shape

    # Calculate sizes in samples
    segment_size_samples = int(sample_rate * segment_size) + extra_samples
    overlap_samples = int(sample_rate * overlap) + extra_samples
    min_segment_samples = int(sample_rate * min_segment_size)
    step_size = segment_size_samples - overlap_samples

    logger.info(
        f"Segmenting audio: total_samples={total_samples}, "
        f"segment_size={segment_size_samples}, "
        f"overlap={overlap_samples}, "
        f"step_size={step_size}"
    )

    segments = []
    segment_idx = 0
    offset = 0

    while offset < total_samples:
        # Calculate end position for this segment
        end = min(offset + segment_size_samples, total_samples)
        segment_length = end - offset

        # Skip if remaining audio is too short
        if segment_length < min_segment_samples:
            logger.debug(
                f"Skipping final segment of {segment_length} samples "
                f"(< {min_segment_samples} min)"
            )
            break

        # Extract segment
        segment_waveform = waveform[:, offset:end]

        # Squeeze channel dimension if mono
        if num_channels == 1:
            segment_waveform = segment_waveform.squeeze(0)

        segment = AudioSegment(
            waveform=segment_waveform,
            sample_rate=sample_rate,
            offset_samples=offset,
            length_samples=segment_length,
            segment_index=segment_idx,
        )
        segments.append(segment)

        segment_idx += 1
        offset += step_size

        # If we've reached or passed the end, stop
        if end >= total_samples:
            break

    logger.info(f"Created {len(segments)} segments")

    return SegmentationResult(
        segments=segments,
        original_duration_samples=total_samples,
        original_duration_seconds=total_samples / sample_rate,
        sample_rate=sample_rate,
        segment_size_samples=segment_size_samples,
        overlap_samples=overlap_samples,
        num_segments=len(segments),
    )

"""
Base classes for stitching utilities.

Defines the interfaces and data structures for combining segment alignments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlignedToken:
    """
    A single aligned token (phone/character).

    This is the basic unit of alignment output from segment-level alignment.
    """
    token_id: Any  # Can be int (ID) or str (symbol)
    timestamp: int  # Frame index
    score: float = 1.0  # Confidence score
    attr: Dict[str, Any] = field(default_factory=dict)  # Extra attributes

    @property
    def has_word_index(self) -> bool:
        """Check if this token has word index information."""
        return "wid" in self.attr


@dataclass
class SegmentAlignment:
    """
    Alignment result for a single audio segment.

    Contains the tokens aligned in this segment along with metadata
    about the segment's position in the original audio.

    Attributes:
        tokens: List of AlignedToken from this segment
        segment_index: Index of this segment (0, 1, 2, ...)
        frame_offset: Frame offset from start of full audio
        sample_offset: Sample offset from start of full audio
        rejected: Whether this segment was rejected by quality checks
        metadata: Additional segment-level information
    """
    tokens: List[AlignedToken]
    segment_index: int = 0
    frame_offset: int = 0
    sample_offset: int = 0
    rejected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def get_word_indices(self) -> List[int]:
        """Extract word indices from tokens that have them."""
        return [t.attr["wid"] for t in self.tokens if "wid" in t.attr]

    def get_tokens_with_word_index(self) -> List[AlignedToken]:
        """Get only tokens that have word index."""
        return [t for t in self.tokens if "wid" in t.attr]


@dataclass
class StitchingResult:
    """
    Result of stitching segment alignments.

    Attributes:
        tokens: Final list of aligned tokens (globally consistent)
        unaligned_regions: List of (start_word_idx, end_word_idx) for gaps
        segment_mapping: Which segment each token came from
        metadata: Stitching statistics and debug info
    """
    tokens: List[AlignedToken]
    unaligned_regions: List[Tuple[int, int]] = field(default_factory=list)
    segment_mapping: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_aligned_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_unaligned_regions(self) -> int:
        return len(self.unaligned_regions)

    def get_word_indices(self) -> List[int]:
        """Get word indices from stitched tokens."""
        return [t.attr["wid"] for t in self.tokens if "wid" in t.attr]


@dataclass
class StitchingConfig:
    """
    Configuration for stitching algorithms.

    Attributes:
        method: Stitching method ("lis", "edit_distance", "diff")

        # LIS parameters
        neighborhood_size: Window for isolated word removal (default: 5)
        outlier_scan_range: Range for outlier detection (default: 3)
        outlier_threshold: Max jump to consider outlier (default: 50)

        # Edit distance parameters
        insertion_cost: Cost for insertion (default: 1)
        deletion_cost: Cost for deletion (default: 1)
        substitution_cost: Cost for substitution (default: 1)

        # General
        frame_duration: Frame duration in seconds (default: 0.02)
        min_confidence: Minimum confidence to keep token (default: 0.0)
    """
    method: str = "lis"

    # LIS parameters
    neighborhood_size: int = 5
    outlier_scan_range: int = 3
    outlier_threshold: int = 50

    # Edit distance parameters
    insertion_cost: float = 1.0
    deletion_cost: float = 1.0
    substitution_cost: float = 1.0

    # General
    frame_duration: float = 0.02
    min_confidence: float = 0.0


class StitcherBackend(ABC):
    """
    Abstract base class for stitching algorithms.

    Stitchers take a list of SegmentAlignment objects and produce
    a single StitchingResult with globally consistent alignment.
    """

    METHOD_NAME: str = "base"

    def __init__(self, config: Optional[StitchingConfig] = None):
        self.config = config or StitchingConfig()

    @property
    def name(self) -> str:
        return self.METHOD_NAME

    @abstractmethod
    def stitch(
        self,
        segment_alignments: List[SegmentAlignment],
        **kwargs,
    ) -> StitchingResult:
        """
        Stitch segment alignments into global alignment.

        Args:
            segment_alignments: List of SegmentAlignment from each segment
            **kwargs: Method-specific parameters

        Returns:
            StitchingResult with globally consistent alignment
        """
        raise NotImplementedError

    def _flatten_tokens(
        self,
        segment_alignments: List[SegmentAlignment],
    ) -> Tuple[List[AlignedToken], List[int]]:
        """
        Flatten all tokens from segments into single list.

        Returns:
            (all_tokens, segment_indices) - tokens and which segment each came from
        """
        all_tokens = []
        segment_indices = []

        for seg in segment_alignments:
            if seg.rejected:
                continue
            for token in seg.tokens:
                all_tokens.append(token)
                segment_indices.append(seg.segment_index)

        return all_tokens, segment_indices

    def _validate_input(self, segment_alignments: List[SegmentAlignment]) -> None:
        """Validate input segment alignments."""
        if not segment_alignments:
            raise ValueError("Empty segment_alignments list")

        # Check for required attributes based on method
        if self.METHOD_NAME == "lis":
            has_word_indices = any(
                t.has_word_index
                for seg in segment_alignments
                for t in seg.tokens
            )
            if not has_word_indices:
                logger.warning(
                    "No word indices found in tokens. "
                    "LIS stitching works best with word index output labels."
                )

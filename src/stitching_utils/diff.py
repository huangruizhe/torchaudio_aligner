"""
Diff-based stitching (Gentle aligner style).

This algorithm uses Python's difflib to align predicted words to
reference transcript, following the approach from Gentle aligner.

Best for:
- Mostly accurate transcripts with minor errors
- Quick alignment without word indices
- When edit distance is too slow

Reference:
- Gentle: https://github.com/strob/gentle/blob/master/gentle/diff_align.py
"""

from typing import List, Tuple, Optional, Dict, Any, Iterator
import difflib
import logging

from .base import (
    StitcherBackend,
    StitchingConfig,
    StitchingResult,
    SegmentAlignment,
    AlignedToken,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Gentle-style diff utilities
# =============================================================================

def word_diff(a: List[str], b: List[str]) -> Iterator[Tuple[str, int, int]]:
    """
    Like difflib.SequenceMatcher but yields one word at a time.

    Following Gentle's approach:
    https://github.com/strob/gentle/blob/master/gentle/diff_align.py

    Args:
        a: Hypothesis/predicted words
        b: Reference words

    Yields:
        (operation, index_in_a, index_in_b) for each word
        Operations: 'equal', 'delete', 'insert', 'replace'
    """
    matcher = difflib.SequenceMatcher(a=a, b=b)
    for op, a_idx, _, b_idx, _ in _by_word(matcher.get_opcodes()):
        yield (op, a_idx, b_idx)


def _by_word(opcodes: List[Tuple]) -> Iterator[Tuple]:
    """
    Take difflib.SequenceMatcher.get_opcodes() output and
    return an equivalent opcode sequence that only modifies
    one word at a time.

    From Gentle:
    https://github.com/strob/gentle/blob/master/gentle/diff_align.py
    """
    for op, s1, e1, s2, e2 in opcodes:
        if op == 'delete':
            for i in range(s1, e1):
                yield (op, i, i + 1, s2, s2)
        elif op == 'insert':
            for i in range(s2, e2):
                yield (op, s1, s1, i, i + 1)
        else:
            # 'equal' or 'replace'
            len1 = e1 - s1
            len2 = e2 - s2
            for i1, i2 in zip(range(s1, e1), range(s2, e2)):
                yield (op, i1, i1 + 1, i2, i2 + 1)
            # Handle length mismatch
            if len1 > len2:
                for i in range(s1 + len2, e1):
                    yield ('delete', i, i + 1, e2, e2)
            if len2 > len1:
                for i in range(s2 + len1, e2):
                    yield ('insert', s1, s1, i, i + 1)


class DiffStitcher(StitcherBackend):
    """
    Diff-based stitcher using Python's difflib.

    Uses SequenceMatcher to find matching blocks between
    predicted and reference words.

    Example:
        >>> stitcher = DiffStitcher(config)
        >>> result = stitcher.stitch(segment_alignments, reference_words=ref)
    """

    METHOD_NAME = "diff"

    def stitch(
        self,
        segment_alignments: List[SegmentAlignment],
        reference_words: Optional[List[str]] = None,
        **kwargs,
    ) -> StitchingResult:
        """
        Stitch segment alignments using diff algorithm.

        Args:
            segment_alignments: List of SegmentAlignment from each segment
            reference_words: Reference transcript as word list
            **kwargs: Additional parameters

        Returns:
            StitchingResult with aligned timestamps
        """
        self._validate_input(segment_alignments)

        # Step 1: Extract predicted words with timestamps
        logger.info("Step 1: Extracting predicted words from segments")
        predicted_words, predicted_tokens = self._extract_words_from_segments(
            segment_alignments
        )

        if not predicted_words:
            logger.warning("No words extracted from segment alignments")
            return StitchingResult(
                tokens=[],
                unaligned_regions=[],
                metadata={"method": self.METHOD_NAME, "error": "no_predicted_words"},
            )

        logger.info(f"Extracted {len(predicted_words)} predicted words")

        if reference_words is None:
            # No reference, dedupe by position
            logger.info("No reference provided, deduping by position")
            return self._dedupe_by_position(segment_alignments, predicted_tokens)

        logger.info(f"Reference has {len(reference_words)} words")

        # Step 2: Compute diff alignment
        logger.info("Step 2: Computing diff alignment")
        alignment = self._compute_diff_alignment(
            predicted_words,
            reference_words,
        )

        # Step 3: Transfer timestamps
        logger.info("Step 3: Transferring timestamps to reference")
        aligned_tokens, unaligned_regions = self._transfer_timestamps(
            alignment,
            predicted_tokens,
            reference_words,
        )

        logger.info(
            f"Stitching complete: {len(aligned_tokens)} tokens, "
            f"{len(unaligned_regions)} unaligned regions"
        )

        return StitchingResult(
            tokens=aligned_tokens,
            unaligned_regions=unaligned_regions,
            metadata={
                "method": self.METHOD_NAME,
                "predicted_words": len(predicted_words),
                "reference_words": len(reference_words),
                "ratio": alignment["ratio"],
                "stats": alignment["stats"],
            },
        )

    def _extract_words_from_segments(
        self,
        segment_alignments: List[SegmentAlignment],
    ) -> Tuple[List[str], List[AlignedToken]]:
        """
        Extract words and their tokens from segments.

        Returns:
            (word_list, token_list)
        """
        words = []
        tokens = []

        for seg in segment_alignments:
            if seg.rejected:
                continue

            for token in seg.tokens:
                word = token.attr.get("word", None)
                if word is None and "tk" in token.attr:
                    word = str(token.attr["tk"])

                if word:
                    words.append(word.lower())
                    tokens.append(token)

        return words, tokens

    def _compute_diff_alignment(
        self,
        predicted: List[str],
        reference: List[str],
    ) -> Dict[str, Any]:
        """
        Compute diff-based alignment using Gentle's word_diff approach.

        Returns dict with:
            - operations: List of (op, pred_idx, ref_idx) tuples
            - pred_to_ref: Dict mapping predicted index to reference index (for 'equal' ops)
        """
        # Normalize for matching
        pred_lower = [w.lower() for w in predicted]
        ref_lower = [w.lower() for w in reference]

        # Use Gentle-style word diff
        operations = list(word_diff(pred_lower, ref_lower))

        # Build mapping from predicted to reference for matches
        pred_to_ref = {}
        num_equal = 0
        num_replace = 0
        num_insert = 0
        num_delete = 0

        for op, pred_idx, ref_idx in operations:
            if op == 'equal':
                pred_to_ref[pred_idx] = ref_idx
                num_equal += 1
            elif op == 'replace':
                # For replace, we can still transfer timestamp with lower confidence
                pred_to_ref[pred_idx] = (ref_idx, 'replace')
                num_replace += 1
            elif op == 'insert':
                num_insert += 1
            elif op == 'delete':
                num_delete += 1

        # Compute ratio
        matcher = difflib.SequenceMatcher(None, pred_lower, ref_lower)

        return {
            "operations": operations,
            "pred_to_ref": pred_to_ref,
            "ratio": matcher.ratio(),
            "stats": {
                "equal": num_equal,
                "replace": num_replace,
                "insert": num_insert,
                "delete": num_delete,
            },
        }

    def _transfer_timestamps(
        self,
        alignment: Dict[str, Any],
        predicted_tokens: List[AlignedToken],
        reference_words: List[str],
    ) -> Tuple[List[AlignedToken], List[Tuple[int, int]]]:
        """
        Transfer timestamps from predicted to reference.

        Follows Gentle's approach:
        - 'equal': Transfer timestamp with full confidence
        - 'replace': Transfer timestamp with reduced confidence (substitution)
        - 'insert': Reference word not found in audio (no timestamp)
        - 'delete': Predicted word not in reference (ignored)

        Returns:
            (aligned_tokens, unaligned_regions)
        """
        pred_to_ref = alignment["pred_to_ref"]

        aligned_tokens = []
        aligned_ref_indices = set()

        for pred_idx, token in enumerate(predicted_tokens):
            if pred_idx in pred_to_ref:
                mapping = pred_to_ref[pred_idx]

                # Check if it's an equal match or replace
                if isinstance(mapping, tuple):
                    ref_idx, op_type = mapping
                    is_replace = (op_type == 'replace')
                else:
                    ref_idx = mapping
                    is_replace = False

                new_token = AlignedToken(
                    token_id=token.token_id,
                    timestamp=token.timestamp,
                    score=token.score * (0.5 if is_replace else 1.0),
                    attr={
                        **token.attr,
                        "wid": ref_idx,
                        "word": reference_words[ref_idx],
                        "aligned_from": pred_idx,
                        "case": "replace" if is_replace else "equal",
                    },
                )
                aligned_tokens.append(new_token)
                aligned_ref_indices.add(ref_idx)

        # Sort by reference index
        aligned_tokens.sort(key=lambda t: t.attr.get("wid", 0))

        # Find unaligned regions (reference words not found in audio)
        unaligned_regions = []
        if aligned_ref_indices:
            rg_min = min(aligned_ref_indices)
            rg_max = max(aligned_ref_indices)

            from .lis import find_unaligned_regions
            unaligned_regions = find_unaligned_regions(
                rg_min, rg_max, aligned_ref_indices
            )

        return aligned_tokens, unaligned_regions

    def _dedupe_by_position(
        self,
        segment_alignments: List[SegmentAlignment],
        predicted_tokens: List[AlignedToken],
    ) -> StitchingResult:
        """
        Deduplicate by keeping unique timestamps.
        """
        seen = {}
        for token in predicted_tokens:
            ts = token.timestamp
            if ts not in seen:
                seen[ts] = token

        sorted_timestamps = sorted(seen.keys())
        tokens = [seen[ts] for ts in sorted_timestamps]

        return StitchingResult(
            tokens=tokens,
            unaligned_regions=[],
            metadata={"method": self.METHOD_NAME, "mode": "dedupe"},
        )

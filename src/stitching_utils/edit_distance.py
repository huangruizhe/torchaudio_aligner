"""
Edit Distance (Levenshtein) based stitching.

This algorithm aligns the predicted text from segments to the reference
transcript using dynamic programming edit distance.

Complexity: O(N*M) where N = predicted length, M = reference length

Best for:
- ASR output alignment (transcribe + timestamp)
- MFA/Gentle results matching to noisy reference
- When word indices are not available

Note: For very long texts (>10K words), consider using LIS instead.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging

from .base import (
    StitcherBackend,
    StitchingConfig,
    StitchingResult,
    SegmentAlignment,
    AlignedToken,
)

logger = logging.getLogger(__name__)


class EditDistanceStitcher(StitcherBackend):
    """
    Edit distance based stitcher for combining segment alignments.

    Aligns predicted words to reference transcript using Levenshtein
    distance, then transfers timestamps.

    Example:
        >>> stitcher = EditDistanceStitcher(config)
        >>> result = stitcher.stitch(segment_alignments, reference_words=ref)
    """

    METHOD_NAME = "edit_distance"

    def stitch(
        self,
        segment_alignments: List[SegmentAlignment],
        reference_words: Optional[List[str]] = None,
        **kwargs,
    ) -> StitchingResult:
        """
        Stitch segment alignments using edit distance.

        Args:
            segment_alignments: List of SegmentAlignment from each segment
            reference_words: Reference transcript as word list
            **kwargs: Override config parameters

        Returns:
            StitchingResult with aligned timestamps
        """
        self._validate_input(segment_alignments)

        # Get costs from config
        ins_cost = kwargs.get("insertion_cost", self.config.insertion_cost)
        del_cost = kwargs.get("deletion_cost", self.config.deletion_cost)
        sub_cost = kwargs.get("substitution_cost", self.config.substitution_cost)

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
            # No reference, just dedupe overlapping segments
            logger.info("No reference provided, deduping by position")
            return self._dedupe_by_position(segment_alignments)

        logger.info(f"Reference has {len(reference_words)} words")

        # Step 2: Compute edit distance alignment
        logger.info("Step 2: Computing edit distance alignment")
        alignment = self._compute_alignment(
            predicted_words,
            reference_words,
            ins_cost=ins_cost,
            del_cost=del_cost,
            sub_cost=sub_cost,
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
                "edit_distance": alignment.get("distance", -1),
            },
        )

    def _extract_words_from_segments(
        self,
        segment_alignments: List[SegmentAlignment],
    ) -> Tuple[List[str], List[AlignedToken]]:
        """
        Extract words and their first tokens from segments.

        Returns:
            (word_list, token_list) - words and corresponding tokens
        """
        words = []
        tokens = []

        for seg in segment_alignments:
            if seg.rejected:
                continue

            for token in seg.tokens:
                # Look for word attribute
                word = token.attr.get("word", None)
                if word is None and "tk" in token.attr:
                    # Reconstruct word from token
                    word = str(token.attr["tk"])

                if word:
                    words.append(word)
                    tokens.append(token)

        return words, tokens

    def _compute_alignment(
        self,
        predicted: List[str],
        reference: List[str],
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        sub_cost: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute edit distance alignment between predicted and reference.

        Returns dict with:
            - distance: edit distance
            - alignment: list of (op, pred_idx, ref_idx) tuples
        """
        n, m = len(predicted), len(reference)

        # DP table
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]

        # Initialize
        for i in range(n + 1):
            dp[i][0] = i * del_cost
        for j in range(m + 1):
            dp[0][j] = j * ins_cost

        # Fill table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if predicted[i - 1].lower() == reference[j - 1].lower():
                    dp[i][j] = dp[i - 1][j - 1]  # Match
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + del_cost,      # Delete from predicted
                        dp[i][j - 1] + ins_cost,      # Insert into predicted
                        dp[i - 1][j - 1] + sub_cost,  # Substitute
                    )

        # Backtrace
        alignment = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and predicted[i - 1].lower() == reference[j - 1].lower():
                alignment.append(("match", i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                alignment.append(("sub", i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + del_cost:
                alignment.append(("del", i - 1, None))
                i -= 1
            else:
                alignment.append(("ins", None, j - 1))
                j -= 1

        alignment.reverse()

        return {
            "distance": dp[n][m],
            "alignment": alignment,
        }

    def _transfer_timestamps(
        self,
        alignment: Dict[str, Any],
        predicted_tokens: List[AlignedToken],
        reference_words: List[str],
    ) -> Tuple[List[AlignedToken], List[Tuple[int, int]]]:
        """
        Transfer timestamps from predicted to reference based on alignment.

        Returns:
            (aligned_tokens, unaligned_regions)
        """
        aligned_tokens = []
        aligned_ref_indices = set()

        for op, pred_idx, ref_idx in alignment["alignment"]:
            if op == "match" and pred_idx is not None and ref_idx is not None:
                # Transfer timestamp
                token = predicted_tokens[pred_idx]
                new_token = AlignedToken(
                    token_id=token.token_id,
                    timestamp=token.timestamp,
                    score=token.score,
                    attr={
                        **token.attr,
                        "wid": ref_idx,
                        "word": reference_words[ref_idx],
                        "aligned_from": pred_idx,
                    },
                )
                aligned_tokens.append(new_token)
                aligned_ref_indices.add(ref_idx)

            elif op == "sub" and pred_idx is not None and ref_idx is not None:
                # Substitution - still transfer timestamp with lower confidence
                token = predicted_tokens[pred_idx]
                new_token = AlignedToken(
                    token_id=token.token_id,
                    timestamp=token.timestamp,
                    score=token.score * 0.5,  # Lower confidence for substitution
                    attr={
                        **token.attr,
                        "wid": ref_idx,
                        "word": reference_words[ref_idx],
                        "aligned_from": pred_idx,
                        "substitution": True,
                    },
                )
                aligned_tokens.append(new_token)
                aligned_ref_indices.add(ref_idx)

        # Find unaligned regions
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
    ) -> StitchingResult:
        """
        Deduplicate overlapping segments by timestamp position.

        Simple approach: keep token with earliest timestamp for duplicates.
        """
        seen_positions = {}  # timestamp -> token
        segment_mapping = {}

        for seg in segment_alignments:
            if seg.rejected:
                continue

            for token in seg.tokens:
                ts = token.timestamp
                if ts not in seen_positions:
                    seen_positions[ts] = token
                    segment_mapping[ts] = seg.segment_index

        # Sort by timestamp
        sorted_timestamps = sorted(seen_positions.keys())
        tokens = [seen_positions[ts] for ts in sorted_timestamps]
        mapping = [segment_mapping[ts] for ts in sorted_timestamps]

        return StitchingResult(
            tokens=tokens,
            unaligned_regions=[],
            segment_mapping=mapping,
            metadata={"method": self.METHOD_NAME, "mode": "dedupe_by_position"},
        )

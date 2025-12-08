"""
Montreal Forced Aligner (MFA) integration.

MFA is a Kaldi-based forced aligner that provides high-quality
word and phone-level alignments for many languages.

Note: MFA assumes accurate transcripts (not fuzzy alignment).
For noisy transcripts, use the WFST aligner instead.

Installation:
    conda install -c conda-forge montreal-forced-aligner

Or using pip:
    pip install montreal-forced-aligner

See: https://montreal-forced-aligner.readthedocs.io/
"""

import os
import tempfile
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import subprocess
import json

import torch

from .base import (
    AlignerBackend,
    AlignmentConfig,
    AlignmentResult,
    AlignedWord,
    AlignedToken,
)

logger = logging.getLogger(__name__)


class MFAAligner(AlignerBackend):
    """
    Montreal Forced Aligner backend.

    MFA provides high-quality forced alignment using acoustic models
    trained with Kaldi. It supports many languages and can be used
    with custom acoustic models and pronunciation dictionaries.

    Note:
        MFA assumes the transcript is accurate. It does NOT support
        fuzzy alignment for noisy transcripts. For that, use WFSTAligner.

    Requirements:
        - MFA installed (conda install -c conda-forge montreal-forced-aligner)
        - Acoustic model for target language
        - Pronunciation dictionary for target language

    Example:
        >>> config = AlignmentConfig(
        ...     backend="mfa",
        ...     language="english_us_arpa",  # MFA model name
        ... )
        >>> aligner = MFAAligner(config)
        >>> result = aligner.align(waveform, text)

    Attributes:
        acoustic_model: MFA acoustic model name or path
        dictionary: MFA dictionary name or path
    """

    BACKEND_NAME = "mfa"

    # Pre-defined language models
    SUPPORTED_LANGUAGES = [
        "english_us_arpa",
        "english_uk_arpa",
        "english_mfa",
        "french_mfa",
        "german_mfa",
        "spanish_mfa",
        "mandarin_mfa",
        "japanese_mfa",
        # Add more as needed
    ]

    def __init__(
        self,
        config: AlignmentConfig,
        acoustic_model: Optional[str] = None,
        dictionary: Optional[str] = None,
    ):
        """
        Args:
            config: AlignmentConfig
            acoustic_model: MFA acoustic model name or path (default: from language)
            dictionary: MFA dictionary name or path (default: from language)
        """
        super().__init__(config)
        self.acoustic_model = acoustic_model or config.language or "english_us_arpa"
        self.dictionary = dictionary or config.language or "english_us_arpa"
        self._mfa_available = None

    def _check_mfa_available(self) -> bool:
        """Check if MFA is installed and available."""
        if self._mfa_available is not None:
            return self._mfa_available

        try:
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                text=True,
            )
            self._mfa_available = result.returncode == 0
            if self._mfa_available:
                logger.info(f"MFA version: {result.stdout.strip()}")
        except FileNotFoundError:
            self._mfa_available = False

        if not self._mfa_available:
            logger.warning(
                "MFA not found. Install with: "
                "conda install -c conda-forge montreal-forced-aligner"
            )

        return self._mfa_available

    def load(self):
        """Check MFA availability and download models if needed."""
        if not self._check_mfa_available():
            raise RuntimeError(
                "Montreal Forced Aligner not installed. "
                "Install with: conda install -c conda-forge montreal-forced-aligner"
            )

        # Check/download acoustic model
        self._ensure_model_downloaded("acoustic", self.acoustic_model)
        self._ensure_model_downloaded("dictionary", self.dictionary)

        self._loaded = True

    def _ensure_model_downloaded(self, model_type: str, model_name: str):
        """Download MFA model if not already present."""
        try:
            # Check if model exists
            result = subprocess.run(
                ["mfa", "model", "list", model_type],
                capture_output=True,
                text=True,
            )

            if model_name not in result.stdout:
                logger.info(f"Downloading MFA {model_type}: {model_name}")
                subprocess.run(
                    ["mfa", "model", "download", model_type, model_name],
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not download {model_type} {model_name}: {e}")

    def align(
        self,
        waveform: torch.Tensor,
        text: str,
        **kwargs,
    ) -> AlignmentResult:
        """
        Align audio to text using MFA.

        Args:
            waveform: Audio tensor (samples,) or (1, samples)
            text: Transcript text (must be accurate)
            **kwargs: Additional options

        Returns:
            AlignmentResult with word and phone alignments
        """
        if not self._loaded:
            self.load()

        # Ensure 1D waveform
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0).squeeze(-1)

        # Create temporary directory for MFA
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            corpus_dir = tmpdir / "corpus"
            output_dir = tmpdir / "output"
            corpus_dir.mkdir()
            output_dir.mkdir()

            # Save audio file
            audio_path = corpus_dir / "audio.wav"
            self._save_audio(waveform, audio_path)

            # Save transcript
            text_path = corpus_dir / "audio.txt"
            text_path.write_text(text)

            # Run MFA alignment
            try:
                subprocess.run(
                    [
                        "mfa", "align",
                        str(corpus_dir),
                        self.dictionary,
                        self.acoustic_model,
                        str(output_dir),
                        "--clean",
                        "--single_speaker",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"MFA alignment failed: {e.stderr}")
                return AlignmentResult(
                    word_alignments={},
                    unaligned_indices=[],
                    metadata={"error": str(e)},
                )

            # Parse TextGrid output
            textgrid_path = output_dir / "audio.TextGrid"
            if not textgrid_path.exists():
                logger.error("MFA did not produce output")
                return AlignmentResult(
                    word_alignments={},
                    unaligned_indices=[],
                    metadata={"error": "No output produced"},
                )

            return self._parse_textgrid(textgrid_path, self.config.frame_duration)

    def _save_audio(self, waveform: torch.Tensor, path: Path):
        """Save waveform to file."""
        try:
            import torchaudio
            torchaudio.save(str(path), waveform.unsqueeze(0), self.config.sample_rate)
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile
            scipy.io.wavfile.write(
                str(path),
                self.config.sample_rate,
                (waveform.numpy() * 32767).astype('int16'),
            )

    def _parse_textgrid(
        self,
        textgrid_path: Path,
        frame_duration: float,
    ) -> AlignmentResult:
        """
        Parse MFA TextGrid output to AlignmentResult.

        Args:
            textgrid_path: Path to TextGrid file
            frame_duration: Duration of one frame in seconds

        Returns:
            AlignmentResult
        """
        try:
            import textgrid
        except ImportError:
            # Simple TextGrid parser
            return self._parse_textgrid_simple(textgrid_path, frame_duration)

        tg = textgrid.TextGrid.fromFile(str(textgrid_path))

        word_alignments = {}
        word_idx = 0

        # Find word tier
        word_tier = None
        phone_tier = None
        for tier in tg.tiers:
            if tier.name.lower() == "words":
                word_tier = tier
            elif tier.name.lower() == "phones":
                phone_tier = tier

        if word_tier is None:
            return AlignmentResult(word_alignments={}, metadata={"error": "No word tier"})

        # Parse words
        for interval in word_tier:
            if interval.mark and interval.mark.strip():
                word_alignments[word_idx] = AlignedWord(
                    word=interval.mark.strip(),
                    start_frame=int(interval.minTime / frame_duration),
                    end_frame=int(interval.maxTime / frame_duration),
                )
                word_idx += 1

        # Add phones if available
        if phone_tier:
            for interval in phone_tier:
                if interval.mark and interval.mark.strip():
                    phone_start = int(interval.minTime / frame_duration)
                    # Find which word this phone belongs to
                    for wid, word in word_alignments.items():
                        if word.start_frame <= phone_start < word.end_frame:
                            word.chars.append(AlignedToken(
                                token_id=interval.mark.strip(),
                                timestamp=phone_start,
                                score=1.0,
                            ))
                            break

        return AlignmentResult(
            word_alignments=word_alignments,
            metadata={"backend": self.BACKEND_NAME},
        )

    def _parse_textgrid_simple(
        self,
        textgrid_path: Path,
        frame_duration: float,
    ) -> AlignmentResult:
        """Simple TextGrid parser without external dependency."""
        content = textgrid_path.read_text()
        word_alignments = {}
        word_idx = 0

        in_word_tier = False
        in_intervals = False

        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if 'name = "words"' in line.lower():
                in_word_tier = True
            elif 'name = "' in line.lower() and in_word_tier:
                in_word_tier = False

            if in_word_tier and 'intervals [' in line:
                # Parse interval
                xmin = xmax = text = None
                i += 1
                while i < len(lines) and 'intervals [' not in lines[i]:
                    l = lines[i].strip()
                    if l.startswith('xmin'):
                        xmin = float(l.split('=')[1].strip())
                    elif l.startswith('xmax'):
                        xmax = float(l.split('=')[1].strip())
                    elif l.startswith('text'):
                        text = l.split('=')[1].strip().strip('"')
                    i += 1

                if text and text.strip():
                    word_alignments[word_idx] = AlignedWord(
                        word=text.strip(),
                        start_frame=int(xmin / frame_duration) if xmin else 0,
                        end_frame=int(xmax / frame_duration) if xmax else 0,
                    )
                    word_idx += 1
                continue

            i += 1

        return AlignmentResult(
            word_alignments=word_alignments,
            metadata={"backend": self.BACKEND_NAME},
        )

    def align_directory(
        self,
        corpus_dir: str,
        output_dir: str,
        num_jobs: int = 4,
    ) -> None:
        """
        Batch align a directory of audio files.

        This is more efficient than aligning files one by one.

        Args:
            corpus_dir: Directory with audio files and transcripts
            output_dir: Output directory for TextGrids
            num_jobs: Number of parallel jobs
        """
        if not self._loaded:
            self.load()

        subprocess.run(
            [
                "mfa", "align",
                corpus_dir,
                self.dictionary,
                self.acoustic_model,
                output_dir,
                "--clean",
                "-j", str(num_jobs),
            ],
            check=True,
        )

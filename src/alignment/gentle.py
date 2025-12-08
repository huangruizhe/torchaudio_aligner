"""
Gentle aligner integration.

Gentle is an open-source forced alignment tool for English.
It uses Kaldi and provides word and phone-level alignments.

Gentle has limited tolerance for transcript errors and works
best with accurate transcripts.

Installation:
    git clone https://github.com/lowerquality/gentle.git
    cd gentle
    ./install.sh

Or using Docker:
    docker run -p 8765:8765 lowerquality/gentle

Or using pip (for Colab):
    pip install gentle

See: https://github.com/lowerquality/gentle

Modes:
    - Full alignment: align(waveform, text) - aligns entire audio
    - Segment alignment: align_segment(waveform, text) - for short segments
    - Parallel: Uses nthreads for multi-threaded alignment
"""

import os
import tempfile
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
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


class GentleAligner(AlignerBackend):
    """
    Gentle aligner backend (English only).

    Gentle provides forced alignment for English using Kaldi.
    It can be used via:
    1. Python API (if gentle is installed)
    2. REST API (if gentle server is running)

    Note:
        Gentle is English-only and assumes relatively accurate
        transcripts. For noisy transcripts or other languages,
        use WFSTAligner instead.

    Example:
        >>> config = AlignmentConfig(backend="gentle")
        >>> aligner = GentleAligner(config)
        >>> result = aligner.align(waveform, text)

    Attributes:
        server_url: URL of Gentle server (default: http://localhost:8765)
        use_server: If True, use REST API instead of Python API
    """

    BACKEND_NAME = "gentle"
    SUPPORTED_LANGUAGES = ["eng", "en", "english"]

    def __init__(
        self,
        config: AlignmentConfig,
        server_url: str = "http://localhost:8765",
        use_server: Optional[bool] = None,
    ):
        """
        Args:
            config: AlignmentConfig
            server_url: URL of Gentle server
            use_server: Force server mode (auto-detect if None)
        """
        super().__init__(config)
        self.server_url = server_url
        self.use_server = use_server
        self._gentle_available = None
        self._server_available = None

    def _check_gentle_python(self) -> bool:
        """Check if Gentle Python API is available."""
        try:
            from gentle import ForcedAligner
            return True
        except ImportError:
            return False

    def _check_gentle_server(self) -> bool:
        """Check if Gentle server is running."""
        try:
            import requests
            response = requests.get(f"{self.server_url}/", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def load(self):
        """Check Gentle availability."""
        if self.use_server is True:
            # Force server mode
            if not self._check_gentle_server():
                raise RuntimeError(
                    f"Gentle server not available at {self.server_url}. "
                    "Start with: docker run -p 8765:8765 lowerquality/gentle"
                )
            self._server_available = True
        elif self.use_server is False:
            # Force Python mode
            if not self._check_gentle_python():
                raise RuntimeError(
                    "Gentle Python API not available. "
                    "Install from: https://github.com/lowerquality/gentle"
                )
            self._gentle_available = True
        else:
            # Auto-detect
            self._gentle_available = self._check_gentle_python()
            self._server_available = self._check_gentle_server()

            if not self._gentle_available and not self._server_available:
                raise RuntimeError(
                    "Gentle not available. Options:\n"
                    "1. Install Python API: git clone https://github.com/lowerquality/gentle && cd gentle && ./install.sh\n"
                    "2. Start server: docker run -p 8765:8765 lowerquality/gentle"
                )

        self._loaded = True

    def align(
        self,
        waveform: torch.Tensor,
        text: str,
        nthreads: int = 4,
        **kwargs,
    ) -> AlignmentResult:
        """
        Align audio to text using Gentle.

        Args:
            waveform: Audio tensor
            text: Transcript text (must be accurate, English only)
            nthreads: Number of threads for parallel processing (Python API only)
            **kwargs: Additional options

        Returns:
            AlignmentResult
        """
        if not self._loaded:
            self.load()

        # Ensure 1D waveform
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0).squeeze(-1)

        # Choose method
        if self._server_available:
            return self._align_server(waveform, text)
        else:
            return self._align_python(waveform, text, nthreads=nthreads)

    def _align_python(
        self,
        waveform: torch.Tensor,
        text: str,
        nthreads: int = 4,
    ) -> AlignmentResult:
        """
        Align using Gentle Python API.

        Args:
            waveform: Audio tensor
            text: Transcript
            nthreads: Number of threads for parallel processing
        """
        from gentle import ForcedAligner, resampler
        from gentle.resources import Resources

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
            self._save_audio(waveform, Path(audio_path))

        try:
            # Load resources and create aligner
            resources = Resources()
            aligner = ForcedAligner(
                resources,
                text,
                nthreads=nthreads,
            )

            # Resample audio
            with resampler.resampler(audio_path) as audio_file:
                aligner.transcribe(audio_file)

            # Get results
            result_json = aligner.to_json()

        finally:
            os.unlink(audio_path)

        return self._parse_gentle_json(result_json)

    def _align_server(
        self,
        waveform: torch.Tensor,
        text: str,
    ) -> AlignmentResult:
        """Align using Gentle REST API."""
        import requests

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
            self._save_audio(waveform, Path(audio_path))

        try:
            with open(audio_path, "rb") as audio_file:
                response = requests.post(
                    f"{self.server_url}/transcriptions",
                    data={"transcript": text},
                    files={"audio": audio_file},
                    params={"async": "false"},
                )

            if response.status_code != 200:
                logger.error(f"Gentle server error: {response.status_code}")
                return AlignmentResult(
                    word_alignments={},
                    metadata={"error": f"Server error: {response.status_code}"},
                )

            result_json = response.json()

        finally:
            os.unlink(audio_path)

        return self._parse_gentle_json(result_json)

    def _save_audio(self, waveform: torch.Tensor, path: Path):
        """Save waveform to file."""
        try:
            import torchaudio
            torchaudio.save(str(path), waveform.unsqueeze(0), self.config.sample_rate)
        except ImportError:
            import scipy.io.wavfile
            scipy.io.wavfile.write(
                str(path),
                self.config.sample_rate,
                (waveform.numpy() * 32767).astype('int16'),
            )

    def _parse_gentle_json(self, result: Dict[str, Any]) -> AlignmentResult:
        """
        Parse Gentle JSON output to AlignmentResult.

        Gentle output format:
        {
            "transcript": "...",
            "words": [
                {
                    "word": "hello",
                    "alignedWord": "hello",
                    "case": "success",
                    "start": 0.5,
                    "end": 0.8,
                    "phones": [{"phone": "HH", "duration": 0.1}, ...]
                },
                ...
            ]
        }
        """
        frame_duration = self.config.frame_duration
        word_alignments = {}
        unaligned_indices = []

        for idx, word_data in enumerate(result.get("words", [])):
            case = word_data.get("case", "not-found")

            if case == "success":
                start_frame = int(word_data.get("start", 0) / frame_duration)
                end_frame = int(word_data.get("end", 0) / frame_duration)

                # Parse phones as chars
                chars = []
                phone_time = start_frame
                for phone_data in word_data.get("phones", []):
                    chars.append(AlignedToken(
                        token_id=phone_data.get("phone", ""),
                        timestamp=phone_time,
                        score=1.0,
                    ))
                    phone_time += int(phone_data.get("duration", 0) / frame_duration)

                word_alignments[idx] = AlignedWord(
                    word=word_data.get("word", ""),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    score=1.0,
                )
            else:
                # Not found in audio
                if unaligned_indices and unaligned_indices[-1][1] == idx - 1:
                    # Extend previous unaligned region
                    unaligned_indices[-1] = (unaligned_indices[-1][0], idx)
                else:
                    unaligned_indices.append((idx, idx))

        return AlignmentResult(
            word_alignments=word_alignments,
            unaligned_indices=unaligned_indices,
            metadata={"backend": self.BACKEND_NAME},
        )

    def get_visualization_html(
        self,
        result: AlignmentResult,
        audio_path: str,
        text: str,
        frame_duration: float = 0.02,
    ) -> str:
        """
        Generate Gentle-style HTML visualization.

        Args:
            result: AlignmentResult
            audio_path: Path to audio file
            text: Original transcript
            frame_duration: Frame duration in seconds

        Returns:
            HTML string for visualization
        """
        # Build JSON for visualization
        words = []
        for idx, (wid, word) in enumerate(sorted(result.word_alignments.items())):
            words.append({
                "word": word.word,
                "alignedWord": word.word.lower(),
                "case": "success",
                "start": f"{word.start_seconds(frame_duration):.2f}",
                "end": f"{word.end_seconds(frame_duration):.2f}",
                "phones": [
                    {"phone": p.token_id, "duration": f"{frame_duration:.2f}"}
                    for p in word.chars
                ],
            })

        # Add unaligned words
        text_words = text.split()
        for start, end in result.unaligned_indices:
            for i in range(start, end + 1):
                if i < len(text_words):
                    words.insert(i, {
                        "word": text_words[i],
                        "case": "not-found-in-audio",
                    })

        inline_json = {
            "transcript": text,
            "words": words,
        }

        # Simple HTML template
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Alignment Visualization</title>
    <style>
        .word {{ display: inline-block; margin: 2px; padding: 2px 4px; }}
        .success {{ background-color: #90EE90; }}
        .not-found {{ background-color: #FFB6C1; }}
    </style>
</head>
<body>
    <h1>Alignment Visualization</h1>
    <audio controls src="{audio_path}"></audio>
    <div id="transcript">
        {"".join(
            f'<span class="word {"success" if w.get("case") == "success" else "not-found"}" '
            f'data-start="{w.get("start", "")}" data-end="{w.get("end", "")}">{w["word"]}</span>'
            for w in words
        )}
    </div>
    <script>
        var INLINE_JSON = {json.dumps(inline_json)};
        // Click to seek functionality
        document.querySelectorAll('.word').forEach(function(el) {{
            el.onclick = function() {{
                var audio = document.querySelector('audio');
                var start = parseFloat(this.getAttribute('data-start'));
                if (!isNaN(start)) {{
                    audio.currentTime = start;
                    audio.play();
                }}
            }};
        }});
    </script>
</body>
</html>"""

        return html

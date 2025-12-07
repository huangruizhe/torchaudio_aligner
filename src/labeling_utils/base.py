"""
Base classes for CTC model backends.

This module defines the abstract interface that all backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class VocabInfo:
    """
    Vocabulary information for a CTC model.

    Attributes:
        labels: List of token labels in vocabulary order
        label_to_id: Token -> ID mapping
        id_to_label: ID -> Token mapping
        blank_id: CTC blank token ID (usually 0)
        unk_id: Unknown token ID (None if not present)
        blank_token: Blank token string representation
        unk_token: Unknown token string representation
    """
    labels: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    blank_id: int = 0
    unk_id: Optional[int] = None
    blank_token: str = "-"
    unk_token: str = "*"

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, token: str) -> bool:
        return token in self.label_to_id


@dataclass
class BackendConfig:
    """
    Configuration for a CTC model backend.

    Attributes:
        model_name: Model identifier (HuggingFace ID, path, or pipeline name)
        language: Target language code (ISO 639-3 for MMS)
        device: Device to run inference on ("cuda", "cpu", "mps")
        dtype: Model dtype (torch.float32, torch.float16, torch.bfloat16)
        with_star: Whether to include <star>/<unk> token in emissions
        trust_remote_code: Whether to trust remote code for HuggingFace models
        cache_dir: Directory to cache downloaded models
        chunk_size: For streaming/chunked inference (optional)
        extra_options: Additional backend-specific options
    """
    model_name: str
    language: Optional[str] = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    with_star: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    chunk_size: Optional[int] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)


class CTCModelBackend(ABC):
    """
    Abstract base class for CTC model backends.

    All backends must implement:
    - load(): Load the model and any processors
    - get_emissions(): Extract frame-wise log posteriors
    - get_vocab_info(): Get vocabulary information

    Optional overrides:
    - get_emissions_batched(): Efficient batch inference
    - supports_language(): Check language support
    - frame_duration: Override default frame duration
    - sample_rate: Override default sample rate

    Example:
        class MyBackend(CTCModelBackend):
            def load(self):
                self._model = load_my_model(self.config.model_name)
                self._loaded = True

            def get_emissions(self, waveform, lengths=None):
                return self._model(waveform)

            def get_vocab_info(self):
                return VocabInfo(...)
    """

    # Backend metadata - subclasses should override
    BACKEND_NAME: str = "base"
    SUPPORTED_MODELS: List[str] = []

    def __init__(self, config: BackendConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._vocab_info: Optional[VocabInfo] = None
        self._loaded = False

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return self.BACKEND_NAME or self.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and processor.

        This method should:
        1. Load the model weights
        2. Load any tokenizer/processor
        3. Move model to the configured device
        4. Set self._loaded = True
        5. Build vocabulary info

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If model configuration is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def get_emissions(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frame-wise log posteriors (emissions) from audio.

        Args:
            waveform: Audio tensor of shape (batch, samples) or (samples,)
            lengths: Optional tensor of actual sample counts per batch item

        Returns:
            emissions: Log posteriors of shape (batch, frames, vocab_size)
            emission_lengths: Number of valid frames for each batch item

        Raises:
            RuntimeError: If model not loaded
        """
        raise NotImplementedError

    @abstractmethod
    def get_vocab_info(self) -> VocabInfo:
        """
        Get vocabulary information for this model.

        Returns:
            VocabInfo containing labels, mappings, and special token IDs

        Raises:
            RuntimeError: If model not loaded
        """
        raise NotImplementedError

    def get_emissions_batched(
        self,
        waveforms: List[torch.Tensor],
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        """
        Batch inference for multiple audio files.

        Default implementation processes one at a time.
        Subclasses can override for more efficient batching with padding.

        Args:
            waveforms: List of 1D audio tensors (variable length)
            batch_size: Maximum batch size for inference

        Returns:
            List of emission tensors, one per input audio
        """
        results = []
        for waveform in waveforms:
            emissions, _ = self.get_emissions(waveform.unsqueeze(0))
            results.append(emissions.squeeze(0))
        return results

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode a sequence of token IDs to text.

        This uses the model's tokenizer to properly convert token IDs
        back to text, handling BPE, word boundaries, etc.

        Args:
            token_ids: List of token IDs (after CTC collapse, blanks removed)

        Returns:
            Decoded text string
        """
        # Default implementation: join labels from vocab_info
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        vocab = self.get_vocab_info()
        tokens = [vocab.id_to_label.get(idx, "") for idx in token_ids]
        # Simple join - subclasses should override for proper tokenizer decode
        return "".join(tokens)

    def greedy_decode(self, emissions: torch.Tensor) -> str:
        """
        Perform greedy CTC decoding on emissions.

        Takes argmax at each frame, collapses repeats, removes blanks,
        then uses the tokenizer to decode to text.

        Args:
            emissions: Log posteriors of shape (frames, vocab_size)

        Returns:
            Decoded text string
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        vocab = self.get_vocab_info()

        # Get most likely token at each frame
        indices = emissions.argmax(dim=-1).tolist()

        # Collapse consecutive duplicates
        collapsed = []
        prev = None
        for idx in indices:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Remove blanks
        token_ids = [idx for idx in collapsed if idx != vocab.blank_id]

        # Use tokenizer to decode
        return self.decode_tokens(token_ids)

    def supports_language(self, language: str) -> bool:
        """
        Check if the model supports a specific language.

        Args:
            language: ISO 639-3 language code (e.g., "eng", "fra", "cmn")

        Returns:
            True if language is supported, False otherwise
        """
        return True  # Default: assume all languages supported

    @property
    def frame_duration(self) -> float:
        """
        Frame duration in seconds.

        Default is 0.02 (20ms) for Wav2Vec2-style models.
        Override if your model uses different frame rates.
        """
        return 0.02

    @property
    def sample_rate(self) -> int:
        """
        Expected input audio sample rate in Hz.

        Default is 16000 Hz for most speech models.
        Override if your model expects different sample rates.
        """
        return 16000

    def unload(self) -> None:
        """
        Unload the model to free memory.

        Call this when done with the model to release GPU memory.
        """
        self._model = None
        self._processor = None
        self._vocab_info = None
        self._loaded = False

        # Try to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model_name!r}, "
            f"loaded={self._loaded})"
        )

    def __enter__(self):
        """Context manager entry - load the model."""
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload the model."""
        self.unload()
        return False

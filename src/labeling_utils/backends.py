"""
CTC Model Backends - Extensible framework for different model sources.

Each backend wraps a specific model API (HuggingFace, TorchAudio, NeMo, etc.)
and provides a unified interface for:
1. Loading the model
2. Extracting frame-wise posteriors (emissions)
3. Getting vocabulary/labels information
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VocabInfo:
    """Vocabulary information for a CTC model."""
    labels: List[str]  # List of token labels
    label_to_id: Dict[str, int]  # Token -> ID mapping
    id_to_label: Dict[int, str]  # ID -> Token mapping
    blank_id: int = 0  # CTC blank token ID (usually 0)
    unk_id: Optional[int] = None  # Unknown token ID
    blank_token: str = "-"  # Blank token string
    unk_token: str = "*"  # Unknown token string


@dataclass
class BackendConfig:
    """Configuration for a CTC model backend."""
    model_name: str
    language: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    # Model-specific options
    with_star: bool = True  # Whether to include <star>/<unk> token
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    # Inference options
    chunk_size: Optional[int] = None  # For streaming/chunked inference
    extra_options: Dict[str, Any] = field(default_factory=dict)


class CTCModelBackend(ABC):
    """
    Abstract base class for CTC model backends.

    All backends must implement:
    - load(): Load the model and processor
    - get_emissions(): Extract frame-wise posteriors
    - get_vocab_info(): Get vocabulary information

    Optional:
    - get_emissions_batched(): Batch inference for efficiency
    - supports_language(): Check if language is supported
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._vocab_info: Optional[VocabInfo] = None
        self._loaded = False

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return self.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
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
            lengths: Optional tensor of actual lengths for each batch item

        Returns:
            emissions: Log posteriors of shape (batch, frames, vocab_size)
            emission_lengths: Number of frames for each batch item
        """
        raise NotImplementedError

    @abstractmethod
    def get_vocab_info(self) -> VocabInfo:
        """Get vocabulary information for this model."""
        raise NotImplementedError

    def get_emissions_batched(
        self,
        waveforms: List[torch.Tensor],
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        """
        Batch inference for multiple audio files.

        Default implementation: process one at a time.
        Subclasses can override for more efficient batching.
        """
        results = []
        for waveform in waveforms:
            emissions, _ = self.get_emissions(waveform.unsqueeze(0))
            results.append(emissions.squeeze(0))
        return results

    def supports_language(self, language: str) -> bool:
        """Check if the model supports a specific language."""
        return True  # Default: assume all languages supported

    @property
    def frame_duration(self) -> float:
        """Frame duration in seconds (typically 0.02 for 20ms)."""
        return 0.02  # Default for Wav2Vec2-style models

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate."""
        return 16000  # Default for most speech models

    def __repr__(self):
        return f"{self.name}(model={self.config.model_name}, loaded={self._loaded})"


class HuggingFaceCTCBackend(CTCModelBackend):
    """
    HuggingFace Transformers backend for CTC models.

    Supports:
    - facebook/mms-1b-all (1100+ languages)
    - facebook/mms-1b-fl102 (102 languages)
    - facebook/wav2vec2-* models
    - Any Wav2Vec2ForCTC compatible model

    Example:
        config = BackendConfig(model_name="facebook/mms-1b-all", language="eng")
        backend = HuggingFaceCTCBackend(config)
        backend.load()
        emissions, lengths = backend.get_emissions(waveform)
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._processor = None
        self._model = None

    def load(self) -> None:
        """Load model and processor from HuggingFace."""
        try:
            from transformers import Wav2Vec2ForCTC, AutoProcessor
        except ImportError:
            raise ImportError(
                "HuggingFace Transformers required. Install with: pip install transformers"
            )

        model_name = self.config.model_name
        language = self.config.language
        device = self.config.device

        logger.info(f"Loading HuggingFace model: {model_name}")

        # Handle MMS models with language adapters
        if "mms" in model_name.lower() and language:
            logger.info(f"Loading with language adapter: {language}")
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                target_lang=language,
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                target_lang=language,
                ignore_mismatched_sizes=True,
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )
        else:
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )

        # Move to device and set dtype
        self._model = self._model.to(device)
        if self.config.dtype == torch.float16 and device != "cpu":
            self._model = self._model.half()

        self._model.eval()
        self._loaded = True

        # Build vocab info
        self._build_vocab_info()

        logger.info(f"Model loaded successfully. Vocab size: {len(self._vocab_info.labels)}")

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from processor."""
        tokenizer = self._processor.tokenizer

        # Get vocabulary
        vocab = tokenizer.get_vocab()
        labels = [""] * len(vocab)
        for token, idx in vocab.items():
            if idx < len(labels):
                labels[idx] = token

        # Find special tokens
        blank_token = tokenizer.pad_token or "<pad>"
        blank_id = vocab.get(blank_token, 0)

        unk_token = tokenizer.unk_token or "<unk>"
        unk_id = vocab.get(unk_token, None)

        # For MMS, the vocab uses '|' as word boundary and specific tokens
        # Normalize to our convention
        self._vocab_info = VocabInfo(
            labels=labels,
            label_to_id=vocab,
            id_to_label={v: k for k, v in vocab.items()},
            blank_id=blank_id,
            unk_id=unk_id,
            blank_token=blank_token,
            unk_token=unk_token,
        )

    def get_vocab_info(self) -> VocabInfo:
        """Get vocabulary information."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._vocab_info

    def get_emissions(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frame-wise log posteriors from audio.

        Args:
            waveform: Audio tensor (batch, samples) or (samples,)
            lengths: Optional lengths tensor

        Returns:
            emissions: Log posteriors (batch, frames, vocab_size)
            emission_lengths: Frame counts per batch item
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]
        device = self.config.device

        # Process through model
        with torch.inference_mode():
            # Prepare inputs
            inputs = self._processor(
                waveform.cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = self._model(
                input_values,
                attention_mask=attention_mask,
            )

            # Get log probabilities
            logits = outputs.logits
            emissions = F.log_softmax(logits, dim=-1)

        # Calculate emission lengths
        # Wav2Vec2 has a specific downsampling ratio
        # Typically: (input_length - kernel_size) / stride + 1 for each conv layer
        # For simplicity, we use the actual output length
        emission_lengths = torch.full(
            (batch_size,),
            emissions.shape[1],
            dtype=torch.long,
            device=emissions.device,
        )

        # If we have input lengths, estimate output lengths
        if lengths is not None:
            # Approximate downsampling ratio (varies by model, ~320 for Wav2Vec2)
            downsample_ratio = waveform.shape[1] / emissions.shape[1]
            emission_lengths = (lengths / downsample_ratio).long()
            emission_lengths = torch.clamp(emission_lengths, max=emissions.shape[1])

        # Add <star>/<unk> dimension if requested and not present
        if self.config.with_star and self._vocab_info.unk_id is None:
            # Append a constant log-prob column for unknown token
            star_dim = torch.full(
                (emissions.shape[0], emissions.shape[1], 1),
                -5.0,  # Low probability for unknown
                device=emissions.device,
                dtype=emissions.dtype,
            )
            emissions = torch.cat([emissions, star_dim], dim=-1)

        return emissions, emission_lengths

    def get_emissions_batched(
        self,
        waveforms: List[torch.Tensor],
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        """Efficient batched emission extraction."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        device = self.config.device

        for i in range(0, len(waveforms), batch_size):
            batch = waveforms[i:i + batch_size]

            # Pad to same length within batch
            max_len = max(w.shape[-1] for w in batch)
            padded = torch.zeros(len(batch), max_len)
            lengths = torch.zeros(len(batch), dtype=torch.long)

            for j, w in enumerate(batch):
                if w.dim() == 2:
                    w = w.squeeze(0)
                padded[j, :w.shape[-1]] = w
                lengths[j] = w.shape[-1]

            # Get emissions
            emissions, emission_lengths = self.get_emissions(padded, lengths)

            # Extract individual results (unpad)
            for j in range(len(batch)):
                result = emissions[j, :emission_lengths[j]]
                results.append(result)

        return results


class TorchAudioPipelineBackend(CTCModelBackend):
    """
    TorchAudio Pipeline backend for CTC models.

    This backend wraps models loaded via torchaudio.pipelines, which have a
    different API: model(waveform, lengths) -> (emissions, emission_lengths)

    Supports:
    - torchaudio.pipelines.MMS_FA (forced alignment)
    - torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    - Custom models from your audio repo branch

    Note: TorchAudio is transitioning to maintenance mode, but this backend
    is needed for models like MMS_FA that aren't on HuggingFace.

    Example:
        config = BackendConfig(model_name="MMS_FA", language="eng")
        backend = TorchAudioPipelineBackend(config)
        backend.load()
        emissions, lengths = backend.get_emissions(waveform)
    """

    # Known pipeline names
    KNOWN_PIPELINES = {
        "MMS_FA": "torchaudio.pipelines.MMS_FA",
        "WAV2VEC2_ASR_BASE_960H": "torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H",
        "WAV2VEC2_ASR_LARGE_960H": "torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H",
    }

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._bundle = None
        self._model = None
        self._labels = None

    def load(self) -> None:
        """Load model from TorchAudio pipeline."""
        try:
            import torchaudio
        except ImportError:
            raise ImportError(
                "TorchAudio required. Install with: pip install torchaudio"
            )

        model_name = self.config.model_name
        device = self.config.device

        logger.info(f"Loading TorchAudio pipeline: {model_name}")

        # Get the pipeline bundle
        if model_name.upper() in self.KNOWN_PIPELINES:
            pipeline_path = self.KNOWN_PIPELINES[model_name.upper()]
            # Dynamically get the pipeline
            parts = pipeline_path.split(".")
            bundle = torchaudio
            for part in parts[1:]:  # Skip 'torchaudio'
                bundle = getattr(bundle, part)
            self._bundle = bundle
        else:
            # Try to get it directly from torchaudio.pipelines
            try:
                self._bundle = getattr(torchaudio.pipelines, model_name)
            except AttributeError:
                raise ValueError(
                    f"Unknown TorchAudio pipeline: {model_name}. "
                    f"Known pipelines: {list(self.KNOWN_PIPELINES.keys())}"
                )

        # Load the model
        # For MMS_FA, we can optionally include the star token
        if hasattr(self._bundle, 'get_model'):
            if 'MMS_FA' in model_name.upper():
                self._model = self._bundle.get_model(
                    with_star=not self.config.with_star  # MMS_FA adds star internally if with_star=False
                ).to(device)
            else:
                self._model = self._bundle.get_model().to(device)
        else:
            raise ValueError(f"Pipeline {model_name} does not have get_model()")

        self._model.eval()

        # Get labels
        if hasattr(self._bundle, 'get_labels'):
            if 'MMS_FA' in model_name.upper() and self.config.with_star:
                self._labels = self._bundle.get_labels(star="*")
            else:
                self._labels = self._bundle.get_labels()
        else:
            self._labels = None

        self._loaded = True
        self._build_vocab_info()

        logger.info(f"Model loaded. Labels: {self._labels[:10] if self._labels else 'N/A'}...")

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from labels."""
        if self._labels is None:
            # Default for unknown models
            self._vocab_info = VocabInfo(
                labels=[],
                label_to_id={},
                id_to_label={},
                blank_id=0,
                unk_id=None,
            )
            return

        labels = list(self._labels)
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for idx, label in enumerate(labels)}

        # Find special tokens
        blank_token = "-"
        blank_id = label_to_id.get(blank_token, 0)

        unk_token = "*"
        unk_id = label_to_id.get(unk_token, None)

        self._vocab_info = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            blank_id=blank_id,
            unk_id=unk_id,
            blank_token=blank_token,
            unk_token=unk_token,
        )

    def get_vocab_info(self) -> VocabInfo:
        """Get vocabulary information."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._vocab_info

    def get_emissions(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frame-wise log posteriors from audio.

        TorchAudio pipeline API: model(waveform, lengths) -> (emissions, lengths)

        Args:
            waveform: Audio tensor (batch, samples) or (samples,)
            lengths: Optional lengths tensor

        Returns:
            emissions: Log posteriors (batch, frames, vocab_size)
            emission_lengths: Frame counts per batch item
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        device = self.config.device

        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]

        # Default lengths if not provided
        if lengths is None:
            lengths = torch.full((batch_size,), waveform.shape[1], dtype=torch.long)

        # Forward pass - TorchAudio pipeline API
        with torch.inference_mode():
            emissions, emission_lengths = self._model(
                waveform.to(device),
                lengths.to(device),
            )

        # Add star dimension if requested and not present
        if self.config.with_star and self._vocab_info.unk_id is None:
            star_dim = torch.full(
                (emissions.shape[0], emissions.shape[1], 1),
                -5.0,  # Low probability for unknown (as in Tutorial.py)
                device=emissions.device,
                dtype=emissions.dtype,
            )
            emissions = torch.cat([emissions, star_dim], dim=-1)

        return emissions, emission_lengths


# Registry of available backends
_BACKENDS: Dict[str, type] = {
    "huggingface": HuggingFaceCTCBackend,
    "hf": HuggingFaceCTCBackend,
    "torchaudio": TorchAudioPipelineBackend,
    "ta": TorchAudioPipelineBackend,
}


def register_backend(name: str, backend_class: type) -> None:
    """Register a new backend."""
    _BACKENDS[name.lower()] = backend_class


def get_backend(name: str = "huggingface") -> type:
    """Get a backend class by name."""
    name = name.lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name]


def list_backends() -> List[str]:
    """List available backend names."""
    return list(_BACKENDS.keys())

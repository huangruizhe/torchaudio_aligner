"""
TorchAudio Pipeline backend for CTC models.

Supports:
- torchaudio.pipelines.MMS_FA (forced alignment)
- torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
- torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
- Custom pipelines from torchaudio

Note: TorchAudio is transitioning to maintenance mode.
For new projects, prefer the HuggingFace backend when possible.
This backend is useful for models like MMS_FA that aren't on HuggingFace.
"""

from typing import Optional, List, Tuple
import logging

import torch

from ..base import CTCModelBackend, VocabInfo, BackendConfig
from ..registry import backend

logger = logging.getLogger(__name__)


@backend("torchaudio", aliases=["ta", "torchaudio_pipeline"])
class TorchAudioPipelineBackend(CTCModelBackend):
    """
    TorchAudio Pipeline backend for CTC models.

    This backend wraps models loaded via torchaudio.pipelines, which have a
    different API: model(waveform, lengths) -> (emissions, emission_lengths)

    Features:
    - Direct model(waveform, lengths) API
    - Support for MMS_FA forced alignment model
    - Automatic star token handling

    Example:
        config = BackendConfig(
            model_name="MMS_FA",
            device="cuda",
            with_star=True,
        )
        backend = TorchAudioPipelineBackend(config)
        backend.load()
        emissions, lengths = backend.get_emissions(waveform)
    """

    BACKEND_NAME = "torchaudio"

    # Known pipeline names mapped to their full paths
    KNOWN_PIPELINES = {
        "MMS_FA": "torchaudio.pipelines.MMS_FA",
        "WAV2VEC2_ASR_BASE_960H": "torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H",
        "WAV2VEC2_ASR_LARGE_960H": "torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H",
        "WAV2VEC2_ASR_LARGE_LV60K_960H": "torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H",
        "HUBERT_ASR_LARGE": "torchaudio.pipelines.HUBERT_ASR_LARGE",
        "HUBERT_ASR_XLARGE": "torchaudio.pipelines.HUBERT_ASR_XLARGE",
    }

    SUPPORTED_MODELS = list(KNOWN_PIPELINES.keys())

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._bundle = None
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
        self._bundle = self._get_pipeline_bundle(model_name, torchaudio)

        # Load the model
        if hasattr(self._bundle, 'get_model'):
            if self._is_mms_fa(model_name):
                # MMS_FA has special with_star parameter
                # with_star=False means the model will NOT include star in output
                # So we pass the opposite of our config
                self._model = self._bundle.get_model(
                    with_star=self.config.with_star
                ).to(device)
            else:
                self._model = self._bundle.get_model().to(device)
        else:
            raise ValueError(f"Pipeline {model_name} does not have get_model()")

        self._model.eval()

        # Get labels
        self._labels = self._get_labels(model_name)

        self._loaded = True
        self._build_vocab_info()

        logger.info(
            f"Model loaded. "
            f"Vocab size: {len(self._labels) if self._labels else 'unknown'}, "
            f"Device: {device}"
        )

    def _get_pipeline_bundle(self, model_name: str, torchaudio):
        """Get the pipeline bundle by name."""
        # Normalize name
        name_upper = model_name.upper().replace("-", "_")

        # Check known pipelines
        if name_upper in self.KNOWN_PIPELINES:
            pipeline_path = self.KNOWN_PIPELINES[name_upper]
            parts = pipeline_path.split(".")
            bundle = torchaudio
            for part in parts[1:]:  # Skip 'torchaudio'
                bundle = getattr(bundle, part)
            return bundle

        # Try to get it directly from torchaudio.pipelines
        try:
            return getattr(torchaudio.pipelines, model_name)
        except AttributeError:
            # Try uppercase version
            try:
                return getattr(torchaudio.pipelines, name_upper)
            except AttributeError:
                raise ValueError(
                    f"Unknown TorchAudio pipeline: {model_name}. "
                    f"Known pipelines: {list(self.KNOWN_PIPELINES.keys())}"
                )

    def _is_mms_fa(self, model_name: str) -> bool:
        """Check if this is the MMS_FA model."""
        return "MMS_FA" in model_name.upper()

    def _get_labels(self, model_name: str) -> Optional[tuple]:
        """Get labels from the pipeline bundle."""
        if not hasattr(self._bundle, 'get_labels'):
            return None

        if self._is_mms_fa(model_name) and self.config.with_star:
            return self._bundle.get_labels(star="*")
        else:
            return self._bundle.get_labels()

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from labels."""
        if self._labels is None:
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
        unk_id = label_to_id.get(unk_token)

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

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        TorchAudio MMS_FA uses romanized phoneme labels (e.g., 'a', 'i', 'n').
        For Wav2Vec2/HuBERT pipelines, uses character labels with '|' for word boundaries.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        vocab = self.get_vocab_info()
        tokens = [vocab.id_to_label.get(idx, "") for idx in token_ids]

        # MMS_FA uses romanized phonemes without explicit word boundaries
        # Wav2Vec2/HuBERT use '|' for word boundaries
        if self._is_mms_fa(self.config.model_name):
            # Romanized phonemes - just join them
            return "".join(tokens)
        else:
            # Character-based with | as word boundary
            text = "".join(t if t != "|" else " " for t in tokens)
            return " ".join(text.split())

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
            lengths = torch.full(
                (batch_size,),
                waveform.shape[1],
                dtype=torch.long,
            )

        # Forward pass - TorchAudio pipeline API
        with torch.inference_mode():
            emissions, emission_lengths = self._model(
                waveform.to(device),
                lengths.to(device),
            )

        # Add star dimension if requested and not already present
        if self.config.with_star and self._vocab_info.unk_id is None:
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
        """Batched emission extraction with padding."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []

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

    @property
    def frame_duration(self) -> float:
        """Frame duration for Wav2Vec2-style models (20ms)."""
        return 0.02

    @property
    def sample_rate(self) -> int:
        """Expected sample rate (16kHz)."""
        return 16000

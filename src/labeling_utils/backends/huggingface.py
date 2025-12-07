"""
HuggingFace Transformers backend for CTC models.

Supports:
- facebook/mms-1b-all (1100+ languages)
- facebook/mms-1b-fl102 (102 languages)
- facebook/mms-300m (300M parameter variant)
- facebook/wav2vec2-base-960h
- facebook/wav2vec2-large-960h-lv60-self
- Any Wav2Vec2ForCTC compatible model on HuggingFace Hub
"""

from typing import Optional, List, Tuple
import logging

import torch
import torch.nn.functional as F

from ..base import CTCModelBackend, VocabInfo, BackendConfig
from ..registry import backend

logger = logging.getLogger(__name__)


@backend("huggingface", aliases=["hf", "transformers"])
class HuggingFaceCTCBackend(CTCModelBackend):
    """
    HuggingFace Transformers backend for CTC models.

    This backend wraps models from the HuggingFace Hub that are compatible
    with Wav2Vec2ForCTC architecture.

    Features:
    - Automatic language adapter loading for MMS models
    - Efficient batched inference with padding
    - Support for float16 inference on GPU

    Example:
        config = BackendConfig(
            model_name="facebook/mms-1b-all",
            language="eng",
            device="cuda",
        )
        backend = HuggingFaceCTCBackend(config)
        backend.load()
        emissions, lengths = backend.get_emissions(waveform)
    """

    BACKEND_NAME = "huggingface"
    SUPPORTED_MODELS = [
        "facebook/mms-1b-all",
        "facebook/mms-1b-fl102",
        "facebook/mms-300m",
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h-lv60-self",
        "MahmoudAshraf/mms-300m-1130-forced-aligner",
    ]

    # MMS language codes (subset - full list has 1100+ languages)
    MMS_LANGUAGES = {
        "eng": "English",
        "fra": "French",
        "deu": "German",
        "spa": "Spanish",
        "ita": "Italian",
        "por": "Portuguese",
        "nld": "Dutch",
        "rus": "Russian",
        "cmn": "Mandarin Chinese",
        "jpn": "Japanese",
        "kor": "Korean",
        "ara": "Arabic",
        "hin": "Hindi",
        "ben": "Bengali",
        "tam": "Tamil",
        "tel": "Telugu",
        "vie": "Vietnamese",
        "tha": "Thai",
        "ind": "Indonesian",
        "tur": "Turkish",
        "pol": "Polish",
        "ukr": "Ukrainian",
        "ces": "Czech",
        "ell": "Greek",
        "heb": "Hebrew",
        "swe": "Swedish",
        "dan": "Danish",
        "nor": "Norwegian",
        "fin": "Finnish",
        "hun": "Hungarian",
        "ron": "Romanian",
        "cat": "Catalan",
    }

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._processor = None
        self._model = None

    def load(self) -> None:
        """Load model and processor from HuggingFace Hub."""
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
        if self._is_mms_model(model_name) and language:
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
        elif self.config.dtype == torch.bfloat16 and device != "cpu":
            self._model = self._model.to(torch.bfloat16)

        self._model.eval()
        self._loaded = True

        # Build vocab info
        self._build_vocab_info()

        logger.info(
            f"Model loaded successfully. "
            f"Vocab size: {len(self._vocab_info.labels)}, "
            f"Device: {device}"
        )

    def _is_mms_model(self, model_name: str) -> bool:
        """Check if this is an MMS model that supports language adapters."""
        model_lower = model_name.lower()
        return "mms" in model_lower and "mms-fa" not in model_lower

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from processor tokenizer."""
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
        unk_id = vocab.get(unk_token)

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
            # Prepare inputs using processor
            inputs = self._processor(
                waveform.cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(device)
            if self.config.dtype == torch.float16:
                input_values = input_values.half()
            elif self.config.dtype == torch.bfloat16:
                input_values = input_values.to(torch.bfloat16)

            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = self._model(
                input_values,
                attention_mask=attention_mask,
            )

            # Get log probabilities
            logits = outputs.logits.float()  # Ensure float32 for log_softmax
            emissions = F.log_softmax(logits, dim=-1)

        # Calculate emission lengths
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
            emission_lengths = (lengths.float() / downsample_ratio).long()
            emission_lengths = torch.clamp(emission_lengths, max=emissions.shape[1])

        # Add <star>/<unk> dimension if requested and not present
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
        """Efficient batched emission extraction with padding."""
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

    def supports_language(self, language: str) -> bool:
        """Check if language is supported."""
        if not self._is_mms_model(self.config.model_name):
            # Non-MMS models typically only support English
            return language.lower() in ("eng", "en", "english")

        # MMS supports 1100+ languages
        # We only have a subset in MMS_LANGUAGES for reference
        return True  # Assume MMS supports any 3-letter code

    @property
    def frame_duration(self) -> float:
        """Frame duration for Wav2Vec2 models (20ms)."""
        return 0.02

    @property
    def sample_rate(self) -> int:
        """Expected sample rate (16kHz)."""
        return 16000

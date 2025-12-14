"""
OmniASR backend for CTC models.

Supports:
- Facebook/Meta Omnilingual ASR CTC models (1600+ languages)
- omniASR_CTC_300M, omniASR_CTC_1B, omniASR_CTC_3B, omniASR_CTC_7B

Reference:
- https://github.com/facebookresearch/omnilingual-asr
- Models are built on fairseq2 and use Wav2Vec2-style architecture

Note:
    This backend requires the omnilingual-asr package:
    pip install omnilingual-asr
"""

from typing import Optional, List, Tuple
import logging

import torch
import torch.nn.functional as F

from ..base import CTCModelBackend, VocabInfo, BackendConfig
from ..registry import backend

logger = logging.getLogger(__name__)


@backend("omniasr", aliases=["omni", "omnilingual"])
class OmniASRBackend(CTCModelBackend):
    """
    OmniASR backend for CTC models.

    This backend wraps Facebook/Meta's Omnilingual ASR models for
    emission extraction. These are large-scale multilingual CTC models
    supporting 1600+ languages.

    Available models:
    - omniASR_CTC_300M: 325M parameters
    - omniASR_CTC_1B: 975M parameters
    - omniASR_CTC_3B: 3.08B parameters
    - omniASR_CTC_7B: 6.5B parameters

    Example:
        config = BackendConfig(
            model_name="omniASR_CTC_1B",
            language="eng_Latn",  # Language code with script
            device="cuda",
        )
        backend = OmniASRBackend(config)
        backend.load()
        emissions, lengths = backend.get_emissions(waveform)

    Note:
        Language codes use format: {iso639-3}_{script}
        Examples: eng_Latn, deu_Latn, cmn_Hans, arb_Arab
    """

    BACKEND_NAME = "omniasr"
    SUPPORTED_MODELS = [
        "omniASR_CTC_300M",
        "omniASR_CTC_1B",
        "omniASR_CTC_3B",
        "omniASR_CTC_7B",
    ]

    # Common language codes (subset - full list has 1600+)
    LANGUAGE_CODES = {
        "eng": "eng_Latn",
        "deu": "deu_Latn",
        "fra": "fra_Latn",
        "spa": "spa_Latn",
        "ita": "ita_Latn",
        "por": "por_Latn",
        "nld": "nld_Latn",
        "rus": "rus_Cyrl",
        "cmn": "cmn_Hans",
        "jpn": "jpn_Jpan",
        "kor": "kor_Hang",
        "ara": "arb_Arab",
        "hin": "hin_Deva",
        "ben": "ben_Beng",
    }

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._pipeline = None
        self._model = None
        self._sp_model = None  # SentencePiece model for vocab
        self._model_dtype = None
        self._device_obj = None

    def load(self) -> None:
        """Load model from OmniASR."""
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError:
            raise ImportError(
                "OmniASR required. Install with: pip install omnilingual-asr"
            )

        model_name = self.config.model_name
        device = self.config.device

        logger.info(f"Loading OmniASR model: {model_name}")

        # Convert device string to torch.device
        self._device_obj = torch.device(device)

        # Create pipeline (handles model loading and setup)
        self._pipeline = ASRInferencePipeline(model_card=model_name, device=self._device_obj)

        # Access the underlying model
        self._model = self._pipeline.model

        # Get model dtype (usually bfloat16)
        self._model_dtype = next(self._model.parameters()).dtype
        logger.info(f"Model dtype: {self._model_dtype}")

        # Access the SentencePiece model for vocabulary
        self._sp_model = self._pipeline.tokenizer._model

        self._loaded = True

        # Build vocab info from SentencePiece model
        self._build_vocab_info()

        logger.info(
            f"Model loaded successfully. "
            f"Vocab size: {len(self._vocab_info.labels)}, "
            f"Device: {device}"
        )

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from OmniASR's SentencePiece tokenizer."""
        sp_model = self._sp_model
        vocab_size = sp_model.vocabulary_size

        # Build labels list using index_to_token
        labels = []
        label_to_id = {}
        for i in range(vocab_size):
            token = sp_model.index_to_token(i)
            labels.append(token)
            label_to_id[token] = i

        # Special tokens from SentencePiece model
        # For OmniASR CTC: blank=0 (<s>), pad=1, eos=2, unk=3
        blank_id = sp_model.bos_idx  # Index 0, used as CTC blank
        unk_id = sp_model.unk_idx    # Index 3

        self._vocab_info = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label={v: k for k, v in label_to_id.items()},
            blank_id=blank_id,
            unk_id=unk_id,
            blank_token=labels[blank_id],
            unk_token=labels[unk_id],
        )

    def get_vocab_info(self) -> VocabInfo:
        """Get vocabulary information."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._vocab_info

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text using the OmniASR tokenizer.

        OmniASR uses character-level SentencePiece tokenization.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Use the pipeline's token_decoder
        try:
            decoder = self._pipeline.tokenizer.create_decoder()
            token_tensor = torch.tensor(token_ids)
            text = decoder(token_tensor)
            return text.strip() if isinstance(text, str) else str(text)
        except Exception:
            pass

        # Fallback: manual decode using vocab_info
        vocab = self.get_vocab_info()
        tokens = [vocab.id_to_label.get(idx, "") for idx in token_ids]
        text = "".join(tokens)
        return text.strip()

    def get_emissions(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frame-wise log posteriors from audio.

        Uses direct model forward call with fairseq2's BatchLayout for
        efficient batched inference.

        Args:
            waveform: Audio tensor (batch, samples) or (samples,)
            lengths: Optional lengths tensor (in samples)

        Returns:
            emissions: Log posteriors (batch, frames, vocab_size)
            emission_lengths: Frame counts per batch item
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from fairseq2.nn import BatchLayout

        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]

        # Move to device and convert to model dtype
        waveform = waveform.to(self._device_obj, dtype=self._model_dtype)

        # Create lengths tensor if not provided
        if lengths is None:
            lengths = torch.full(
                (batch_size,), waveform.shape[1],
                dtype=torch.long, device=self._device_obj
            )
        else:
            lengths = lengths.to(self._device_obj)

        # Create BatchLayout for fairseq2 model
        batch_layout = BatchLayout(
            waveform.shape,
            seq_lens=lengths,
            device=self._device_obj,
        )

        # Forward pass
        self._model.eval()
        with torch.inference_mode():
            logits, output_layout = self._model(waveform, batch_layout)

        # Convert to log probabilities (use float32 for numerical stability)
        emissions = F.log_softmax(logits.float(), dim=-1)

        # Extract output lengths from BatchLayout
        output_seq_lens = output_layout.seq_lens
        emission_lengths = torch.tensor(
            [l.item() if hasattr(l, 'item') else l for l in output_seq_lens],
            dtype=torch.long,
            device=emissions.device,
        )

        return emissions, emission_lengths

    def supports_language(self, language: str) -> bool:
        """
        Check if language is supported.

        OmniASR supports 1600+ languages. Language codes use format:
        {iso639-3}_{script} (e.g., eng_Latn, cmn_Hans)
        """
        # OmniASR supports most languages
        return True

    @property
    def frame_duration(self) -> float:
        """Frame duration for Wav2Vec2-style models (20ms)."""
        return 0.02

    @property
    def sample_rate(self) -> int:
        """Expected sample rate (16kHz)."""
        return 16000

    @staticmethod
    def get_language_code(iso639_3: str) -> str:
        """
        Convert ISO 639-3 code to OmniASR format.

        Args:
            iso639_3: ISO 639-3 language code (e.g., "eng", "cmn")

        Returns:
            OmniASR language code with script (e.g., "eng_Latn")
        """
        return OmniASRBackend.LANGUAGE_CODES.get(
            iso639_3.lower(),
            f"{iso639_3.lower()}_Latn"  # Default to Latin script
        )

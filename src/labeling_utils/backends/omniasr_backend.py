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
        self._encoder = None
        self._ctc_decoder = None

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

        # Create pipeline
        self._pipeline = ASRInferencePipeline(model_card=model_name)

        # Try to access the underlying model components for emission extraction
        # The pipeline wraps a model that should have encoder and CTC head
        try:
            # Access the model from pipeline
            if hasattr(self._pipeline, 'model'):
                model = self._pipeline.model
            elif hasattr(self._pipeline, '_model'):
                model = self._pipeline._model
            else:
                model = None
                logger.warning(
                    "Could not access underlying model. "
                    "Emission extraction may use fallback method."
                )

            if model is not None:
                # Try to find encoder and CTC components
                if hasattr(model, 'encoder'):
                    self._encoder = model.encoder
                if hasattr(model, 'ctc_decoder') or hasattr(model, 'ctc'):
                    self._ctc_decoder = getattr(model, 'ctc_decoder', None) or getattr(model, 'ctc', None)

                # Move to device if possible
                if hasattr(model, 'to'):
                    model.to(device)

        except Exception as e:
            logger.warning(f"Could not extract model components: {e}")

        self._loaded = True

        # Build vocab info
        self._build_vocab_info()

        logger.info(
            f"Model loaded successfully. "
            f"Vocab size: {len(self._vocab_info.labels)}, "
            f"Device: {device}"
        )

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from OmniASR tokenizer."""
        # Try to get vocabulary from pipeline/model
        labels = []
        label_to_id = {}
        blank_id = 0

        try:
            if hasattr(self._pipeline, 'tokenizer'):
                tokenizer = self._pipeline.tokenizer
            elif hasattr(self._pipeline, '_tokenizer'):
                tokenizer = self._pipeline._tokenizer
            else:
                tokenizer = None

            if tokenizer is not None:
                # Get vocab from tokenizer
                if hasattr(tokenizer, 'get_vocab'):
                    vocab = tokenizer.get_vocab()
                    labels = [""] * len(vocab)
                    for token, idx in vocab.items():
                        if idx < len(labels):
                            labels[idx] = token
                    label_to_id = vocab
                elif hasattr(tokenizer, 'vocab'):
                    vocab = tokenizer.vocab
                    labels = list(vocab.keys())
                    label_to_id = vocab

        except Exception as e:
            logger.warning(f"Could not extract vocabulary: {e}")

        # Fallback: create minimal vocab
        if not labels:
            logger.warning("Using placeholder vocabulary")
            labels = ["<blank>", "<unk>"]
            label_to_id = {"<blank>": 0, "<unk>": 1}

        self._vocab_info = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label={v: k for k, v in label_to_id.items()},
            blank_id=blank_id,
            unk_id=label_to_id.get("<unk>"),
            blank_token="<blank>",
            unk_token="<unk>",
        )

    def get_vocab_info(self) -> VocabInfo:
        """Get vocabulary information."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._vocab_info

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text using the OmniASR tokenizer.

        OmniASR uses a tokenizer similar to fairseq2/sentencepiece.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Try to use the pipeline's tokenizer
        try:
            if hasattr(self._pipeline, 'tokenizer'):
                tokenizer = self._pipeline.tokenizer
            elif hasattr(self._pipeline, '_tokenizer'):
                tokenizer = self._pipeline._tokenizer
            else:
                tokenizer = None

            if tokenizer is not None:
                if hasattr(tokenizer, 'decode'):
                    text = tokenizer.decode(token_ids)
                    return text.strip()
                elif hasattr(tokenizer, 'ids_to_text'):
                    text = tokenizer.ids_to_text(token_ids)
                    return text.strip()
        except Exception:
            pass

        # Fallback: manual decode using vocab_info
        vocab = self.get_vocab_info()
        tokens = [vocab.id_to_label.get(idx, "") for idx in token_ids]
        text = "".join(tokens)
        # Handle BPE word boundary marker
        text = text.replace("▁", " ")
        return " ".join(text.split())

    def get_emissions(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frame-wise log posteriors from audio.

        Args:
            waveform: Audio tensor (batch, samples) or (samples,)
            lengths: Optional lengths tensor (in samples)

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

        with torch.inference_mode():
            # Try direct model access for emissions
            if self._encoder is not None and self._ctc_decoder is not None:
                # Direct emission extraction
                waveform = waveform.to(device)

                # Encode
                encoder_out = self._encoder(waveform)

                # Get CTC logits
                if hasattr(encoder_out, 'output'):
                    features = encoder_out.output
                else:
                    features = encoder_out

                logits = self._ctc_decoder(features)
                emissions = F.log_softmax(logits.float(), dim=-1)

                # Calculate lengths
                if lengths is not None:
                    # Approximate downsampling (typically 320x for wav2vec2)
                    downsample_ratio = waveform.shape[1] / emissions.shape[1]
                    emission_lengths = (lengths.float() / downsample_ratio).long()
                    emission_lengths = torch.clamp(emission_lengths, max=emissions.shape[1])
                else:
                    emission_lengths = torch.full(
                        (batch_size,), emissions.shape[1], dtype=torch.long
                    )

            else:
                # Fallback: try to hook into pipeline's forward pass
                # This may not work for all versions
                raise NotImplementedError(
                    "Direct emission extraction not available for this OmniASR version. "
                    "The pipeline API only exposes transcription, not raw emissions. "
                    "Please check if a newer version of omnilingual-asr exposes "
                    "the model's forward method for emission extraction."
                )

        # Add <star>/<unk> dimension if requested and not present
        if self.config.with_star and self._vocab_info.unk_id is None:
            star_dim = torch.full(
                (emissions.shape[0], emissions.shape[1], 1),
                -5.0,
                device=emissions.device,
                dtype=emissions.dtype,
            )
            emissions = torch.cat([emissions, star_dim], dim=-1)

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

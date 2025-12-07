"""
NeMo backend for CTC models.

Supports:
- NVIDIA NeMo CTC models (Conformer-CTC, QuartzNet, etc.)
- Hybrid RNN-T/CTC models (using CTC head)
- Any NeMo ASR model with CTC decoder

Reference:
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html
- https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb
"""

from typing import Optional, List, Tuple
import logging

import torch
import torch.nn.functional as F

from ..base import CTCModelBackend, VocabInfo, BackendConfig
from ..registry import backend

logger = logging.getLogger(__name__)


@backend("nemo", aliases=["nvidia", "nemo-asr"])
class NeMoCTCBackend(CTCModelBackend):
    """
    NeMo backend for CTC models.

    This backend wraps NVIDIA NeMo ASR models for emission extraction.
    Supports both pure CTC models and hybrid RNN-T/CTC models.

    Features:
    - Automatic CTC head selection for hybrid models
    - Support for various model architectures (Conformer, QuartzNet, etc.)
    - BPE and character tokenizers

    Example:
        config = BackendConfig(
            model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
            device="cuda",
        )
        backend = NeMoCTCBackend(config)
        backend.load()
        emissions, lengths = backend.get_emissions(waveform)

    Note:
        In NeMo, the blank label has the maximum id (vocab_size - 1),
        but our interface requires blank_id=0. This backend handles
        the conversion automatically by rolling the logits dimension.
    """

    BACKEND_NAME = "nemo"
    SUPPORTED_MODELS = [
        "nvidia/stt_en_fastconformer_hybrid_large_pc",
        "nvidia/stt_en_conformer_ctc_large",
        "nvidia/stt_en_conformer_ctc_small",
        "nvidia/stt_en_quartznet15x5",
        "nvidia/stt_en_citrinet_1024",
    ]

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._is_hybrid = False
        self._frame_duration = 0.04  # Default for CTC models

    def load(self) -> None:
        """Load model from NeMo."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo required. Install with: pip install nemo_toolkit[asr]"
            )

        model_name = self.config.model_name
        device = self.config.device

        logger.info(f"Loading NeMo model: {model_name}")

        # Determine model type and load
        if "hybrid" in model_name.lower() or "rnnt" in model_name.lower():
            self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                model_name=model_name,
                map_location=device,
            )
            self._is_hybrid = True
            self._frame_duration = 0.08  # Hybrid models use 80ms frames
            # Switch to CTC decoder for emission extraction
            self._model.change_decoding_strategy(decoder_type="ctc")
            logger.info("Loaded hybrid RNN-T/CTC model, using CTC head")
        else:
            # Try CTC BPE model first, then CTC character model
            try:
                self._model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                    model_name=model_name,
                    map_location=device,
                )
            except Exception:
                self._model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                    model_name=model_name,
                    map_location=device,
                )
            self._frame_duration = 0.04  # Standard CTC models use 40ms frames

        self._model = self._model.to(device)
        self._model.eval()
        self._loaded = True

        # Build vocab info
        self._build_vocab_info()

        logger.info(
            f"Model loaded successfully. "
            f"Vocab size: {len(self._vocab_info.labels)}, "
            f"Device: {device}, "
            f"Frame duration: {self._frame_duration}s"
        )

    def _build_vocab_info(self) -> None:
        """Build vocabulary info from NeMo tokenizer."""
        tokenizer = self._model.tokenizer

        # Get vocabulary - NeMo uses sentencepiece for BPE models
        if hasattr(tokenizer, 'tokenizer'):
            # BPE tokenizer (has inner sentencepiece model)
            sp_model = tokenizer.tokenizer
            vocab_size = sp_model.vocab_size()

            # Build labels list
            # Note: In NeMo, blank is at the END (index = vocab_size)
            # We'll handle this by rolling the emissions
            labels = ["<blank>"]  # Our blank at index 0
            for i in range(vocab_size):
                labels.append(sp_model.id_to_piece(i))

            label_to_id = {label: i for i, label in enumerate(labels)}
            id_to_label = {i: label for i, label in enumerate(labels)}

            # Find unk token
            unk_token = "<unk>"
            unk_id = label_to_id.get(unk_token)
        else:
            # Character tokenizer
            vocab = tokenizer.vocab
            labels = ["<blank>"] + list(vocab.keys())
            label_to_id = {label: i for i, label in enumerate(labels)}
            id_to_label = {i: label for i, label in enumerate(labels)}
            unk_token = "<unk>"
            unk_id = label_to_id.get(unk_token)

        self._vocab_info = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            blank_id=0,  # We roll emissions so blank is at 0
            unk_id=unk_id,
            blank_token="<blank>",
            unk_token=unk_token,
        )

    def get_vocab_info(self) -> VocabInfo:
        """Get vocabulary information."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._vocab_info

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text using the NeMo tokenizer.

        NeMo uses sentencepiece BPE tokenization. Token IDs are offset by +1
        because we roll emissions to put blank at index 0.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokenizer = self._model.tokenizer

        # Convert back: our token IDs have +1 offset (blank at 0)
        # NeMo's original IDs are token_id - 1
        original_ids = [tid - 1 for tid in token_ids if tid > 0]

        # Use sentencepiece decode
        if hasattr(tokenizer, 'tokenizer'):
            # BPE tokenizer with inner sentencepiece model
            sp_model = tokenizer.tokenizer
            text = sp_model.decode(original_ids)
        elif hasattr(tokenizer, 'ids_to_text'):
            # NeMo tokenizer interface
            text = tokenizer.ids_to_text(original_ids)
        else:
            # Fallback: manual decode
            vocab = self.get_vocab_info()
            tokens = [vocab.id_to_label.get(idx, "") for idx in token_ids]
            text = "".join(tokens).replace("▁", " ")

        return text.strip()

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
                       Note: blank is rolled to index 0
            emission_lengths: Frame counts per batch item
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            from nemo.collections.asr.parts.mixins import TranscribeConfig
            from nemo.collections.asr.parts.mixins.transcription import (
                InternalTranscribeConfig,
                move_to_device,
            )
        except ImportError:
            raise ImportError("NeMo transcription utilities not available")

        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]
        device = self.config.device

        # Prepare audio as list (NeMo format)
        if lengths is not None:
            audio_list = [waveform[i, :lengths[i]].cpu() for i in range(batch_size)]
        else:
            audio_list = [waveform[i].cpu() for i in range(batch_size)]

        # Set up transcription config
        transcribe_cfg = TranscribeConfig(
            batch_size=batch_size,
            return_hypotheses=False,
            partial_hypothesis=None,
            num_workers=0,
            channel_selector=None,
            augmentor=None,
            verbose=False,
        )
        transcribe_cfg._internal = InternalTranscribeConfig()

        with torch.inference_mode():
            # Initialize transcription
            self._model._transcribe_on_begin(audio_list, transcribe_cfg)

            # Get dataloader
            dataloader = self._model._transcribe_input_processing(audio_list, transcribe_cfg)

            all_emissions = []
            all_lengths = []

            for test_batch in dataloader:
                # Move batch to device
                test_batch = move_to_device(test_batch, device)

                # Forward pass
                model_outputs = self._model._transcribe_forward(test_batch, transcribe_cfg)

                # Get logits
                logits = model_outputs["logits"]

                # Get emission lengths
                if "encoded_len" in model_outputs:
                    emit_lengths = model_outputs["encoded_len"]
                elif "logits_len" in model_outputs:
                    emit_lengths = model_outputs["logits_len"]
                else:
                    emit_lengths = torch.full(
                        (logits.shape[0],), logits.shape[1], dtype=torch.long
                    )

                # Roll dimension: In NeMo, blank is at the END (max id)
                # We need blank at index 0 for our interface
                logits = logits.roll(shifts=1, dims=-1)

                # Apply log_softmax
                emissions = F.log_softmax(logits.float(), dim=-1)

                all_emissions.append(emissions)
                all_lengths.append(emit_lengths)

            # Concatenate batches
            emissions = torch.cat(all_emissions, dim=0)
            emission_lengths = torch.cat(all_lengths, dim=0)

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
        """Check if language is supported."""
        model_lower = self.config.model_name.lower()

        # Check language code in model name
        if "_en_" in model_lower or "english" in model_lower:
            return language.lower() in ("eng", "en", "english")

        # Multilingual models
        if "multilingual" in model_lower:
            return True

        # Default: assume English-only
        return language.lower() in ("eng", "en", "english")

    @property
    def frame_duration(self) -> float:
        """Frame duration depends on model architecture."""
        return self._frame_duration

    @property
    def sample_rate(self) -> int:
        """Expected sample rate (16kHz)."""
        return 16000

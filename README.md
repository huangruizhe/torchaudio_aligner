# torchaudio_aligner

This repo corresponds to this paper "Long-Form Fuzzy Speech-to-Text Alignment for 1000+ Languages" in ASRU 2025

## Project Structure

```
torchaudio_aligner/
├── src/
│   ├── __init__.py              # Main package init
│   ├── text_frontend/           # Text processing module
│   │   ├── __init__.py
│   │   ├── loaders.py           # Text loading (file, URL, PDF, OCR)
│   │   ├── normalization.py     # Text normalization (TN, num2words, MMS)
│   │   ├── romanization.py      # Romanization (uroman, cutlet, CJK)
│   │   ├── tokenizers.py        # Tokenizers (Char, BPE, Phoneme)
│   │   └── frontend.py          # TextFrontend class, prepare_for_alignment
│   └── audio_frontend/          # Audio processing module
│       ├── __init__.py
│       ├── loaders.py           # Audio loading (torchaudio, soundfile)
│       ├── preprocessing.py     # Preprocessing (resample, mono, normalize)
│       ├── segmentation.py      # Segmentation (AudioSegment, SegmentationResult)
│       ├── enhancement.py       # Enhancement (denoising, VAD, vocals)
│       └── frontend.py          # AudioFrontend class, segment_audio
├── tests/
│   ├── test_text_frontend.ipynb # Text frontend tests (Colab-ready)
│   ├── test_audio_frontend.ipynb # Audio frontend tests (Colab-ready)
│   └── test_download_resources.ipynb
└── README.md
```

## Quick Start

### Text Frontend

```python
from src.text_frontend import prepare_for_alignment, TextFrontend

# One-liner for alignment preparation
result = prepare_for_alignment(
    "transcript.pdf",
    language="en",
    expand_numbers=True,
    romanize=False,
)
print(result.normalized_text)
print(result.tokens)  # Ready for alignment

# Or use the class for more control
frontend = TextFrontend()
text = frontend.load("transcript.pdf")
text = frontend.normalize(text, expand_numbers=True)
tokens = frontend.tokenize(text)
```

### Audio Frontend

```python
from src.audio_frontend import AudioFrontend, segment_audio, AudioEnhancement

# Simple segmentation
result = segment_audio("audio.mp3", segment_size=15.0, overlap=2.0)
waveforms, lengths = result.get_waveforms_batched()

# With enhancement (optional)
enhancer = AudioEnhancement()
enhanced = enhancer.enhance("noisy_audio.mp3", denoise_method="noisereduce")

# Full pipeline
frontend = AudioFrontend(target_sample_rate=16000, mono=True)
result = frontend.process("audio.mp3", segment_size=15.0, overlap=2.0)
```

## Features

### Text Frontend
- **Loading**: File, URL, PDF, OCR (scanned PDFs via pytesseract)
- **Text Normalization (TN)**:
  - num2words for 60+ languages
  - Currency ($, €, £, ¥, ₹), percentages, decimals, ordinals
  - Mixed letter-number patterns (COVID19, B2B, 4K)
- **Romanization**:
  - uroman for 1100+ languages
  - cutlet for Japanese morphological analysis
  - CJK character splitting
- **Tokenizers**:
  - CharTokenizer (MMS vocabulary)
  - BPETokenizer (SentencePiece)
  - PhonemeTokenizer (CMUDict + G2P)
- **Word Count Preservation**: All transforms preserve word count for lossless alignment recovery

### Audio Frontend
- **Loading**: torchaudio (primary), soundfile (fallback)
- **Preprocessing**: Resample, mono conversion, peak normalization
- **Segmentation**: Overlapping segments for divide-and-conquer alignment
- **Batching**: GPU-ready tensor batching with padding
- **Enhancement** (optional):
  - noisereduce: Spectral gating (CPU-friendly, recommended)
  - DeepFilterNet: Deep learning 48kHz denoising
  - Resemble Enhance: AI speech enhancement (GPU)
  - Demucs: Vocal extraction from music/noise
  - Silero VAD: Voice Activity Detection
- **Timestamp Recovery**: TimeMappingManager for silence removal recovery

## Installation

```bash
# Core dependencies
pip install torch torchaudio
pip install requests beautifulsoup4 pypdf uroman-python zhon num2words

# Optional for text frontend
pip install sentencepiece  # BPE tokenizer
pip install cmudict g2p_en  # Phoneme tokenizer
pip install cutlet  # Japanese romanization
pip install pytesseract  # OCR for scanned PDFs

# Optional for audio enhancement
pip install noisereduce  # Lightweight, recommended
pip install demucs pyloudnorm  # Vocal extraction
# pip install deepfilternet  # Requires Rust compiler
# pip install resemble-enhance  # Requires torch==2.1.1
```

## Testing

Open the Colab-ready test notebooks:

1. `tests/test_text_frontend.ipynb` - Text processing tests
2. `tests/test_audio_frontend.ipynb` - Audio processing tests

Both notebooks use modular imports from `src/` and can run end-to-end.

## Key Design Principles

1. **Word Count Preservation**: All text transforms preserve word count, enabling lossless recovery of original text from alignment indices.

2. **Modular Architecture**: Each component (loading, normalization, tokenization, etc.) is in its own file for maintainability.

3. **Optional Dependencies**: Heavy libraries (Demucs, DeepFilterNet, etc.) are optional and gracefully degrade.

4. **Timestamp Recovery**: When audio is modified (silence removal, enhancement), TimeMappingManager tracks the mapping back to original timestamps.

## License

See LICENSE file for details.

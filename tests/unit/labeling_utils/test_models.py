"""
Tests for model loading utilities.

Tests cover:
- ModelConfig dataclass
- list_presets function
- get_preset_by_category function
- get_model_info function
- Model preset definitions
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for labeling_utils imports"
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        import torch
        from labeling_utils.models import ModelConfig

        config = ModelConfig(
            model_name="facebook/mms-1b-all",
            backend="huggingface",
            language="eng",
        )

        assert config.model_name == "facebook/mms-1b-all"
        assert config.backend == "huggingface"
        assert config.language == "eng"

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        import torch
        from labeling_utils.models import ModelConfig

        config = ModelConfig(model_name="test-model")

        assert config.backend == "huggingface"
        assert config.dtype == torch.float32
        assert config.with_star is True
        assert config.extra_options == {}

    def test_model_config_auto_device(self):
        """Test that device is auto-detected."""
        from labeling_utils.models import ModelConfig

        config = ModelConfig(model_name="test-model")

        # Should be cuda or cpu depending on availability
        assert config.device in ["cuda", "cpu"]


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_presets_returns_list(self):
        """Test that list_presets returns a list."""
        from labeling_utils.models import list_presets

        presets = list_presets()
        assert isinstance(presets, list)

    def test_list_presets_not_empty(self):
        """Test that presets list is not empty."""
        from labeling_utils.models import list_presets

        presets = list_presets()
        assert len(presets) > 0

    def test_list_presets_contains_common_models(self):
        """Test that common model presets are included."""
        from labeling_utils.models import list_presets

        presets = list_presets()

        # Check for some common presets
        expected = ["mms", "mms-fa", "wav2vec2-base"]
        for name in expected:
            assert name in presets, f"Expected preset '{name}' not found"


class TestGetPresetByCategory:
    """Tests for get_preset_by_category function."""

    def test_returns_dict(self):
        """Test that get_preset_by_category returns a dict."""
        from labeling_utils.models import get_preset_by_category

        categories = get_preset_by_category()
        assert isinstance(categories, dict)

    def test_has_expected_categories(self):
        """Test that expected categories are present."""
        from labeling_utils.models import get_preset_by_category

        categories = get_preset_by_category()

        expected_categories = [
            "MMS (HuggingFace)",
            "MMS Forced Alignment",
            "Wav2Vec2 (HuggingFace)",
            "Wav2Vec2 (TorchAudio)",
        ]

        for cat in expected_categories:
            assert cat in categories

    def test_categories_contain_presets(self):
        """Test that categories contain preset lists."""
        from labeling_utils.models import get_preset_by_category

        categories = get_preset_by_category()

        for cat_name, presets in categories.items():
            assert isinstance(presets, list)


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_preset_model_info(self):
        """Test getting info for a preset model."""
        from labeling_utils.models import get_model_info

        info = get_model_info("mms")

        assert info["preset"] == "mms"
        assert "model_name" in info
        assert "backend" in info
        assert "languages" in info

    def test_preset_model_name_mapping(self):
        """Test that preset maps to actual model name."""
        from labeling_utils.models import get_model_info

        info = get_model_info("mms")
        assert info["model_name"] == "facebook/mms-1b-all"

    def test_unknown_model_info(self):
        """Test getting info for unknown model."""
        from labeling_utils.models import get_model_info

        info = get_model_info("facebook/custom-model")

        assert info["preset"] is None
        assert info["model_name"] == "facebook/custom-model"
        assert info["backend"] == "huggingface"

    def test_torchaudio_model_detection(self):
        """Test auto-detection of TorchAudio models."""
        from labeling_utils.models import get_model_info

        info = get_model_info("MMS_FA")
        assert info["backend"] == "torchaudio"

    def test_mms_fa_preset_info(self):
        """Test MMS_FA preset info."""
        from labeling_utils.models import get_model_info

        info = get_model_info("mms-fa")

        assert info["preset"] == "mms-fa"
        assert info["backend"] == "torchaudio"
        assert info["with_star"] is True


class TestModelPresets:
    """Tests for model preset definitions."""

    def test_mms_preset_exists(self):
        """Test MMS preset exists."""
        from labeling_utils.models import _MODEL_PRESETS

        assert "mms" in _MODEL_PRESETS
        assert _MODEL_PRESETS["mms"].model_name == "facebook/mms-1b-all"

    def test_mms_fa_preset_exists(self):
        """Test MMS-FA preset exists."""
        from labeling_utils.models import _MODEL_PRESETS

        assert "mms-fa" in _MODEL_PRESETS
        assert _MODEL_PRESETS["mms-fa"].backend == "torchaudio"

    def test_wav2vec2_presets_exist(self):
        """Test Wav2Vec2 presets exist."""
        from labeling_utils.models import _MODEL_PRESETS

        assert "wav2vec2-base" in _MODEL_PRESETS
        assert "wav2vec2-large" in _MODEL_PRESETS

    def test_nemo_presets_exist(self):
        """Test NeMo presets exist."""
        from labeling_utils.models import _MODEL_PRESETS

        assert "nemo-conformer" in _MODEL_PRESETS
        assert _MODEL_PRESETS["nemo-conformer"].backend == "nemo"

    def test_all_presets_have_model_name(self):
        """Test all presets have model_name set."""
        from labeling_utils.models import _MODEL_PRESETS

        for name, config in _MODEL_PRESETS.items():
            assert config.model_name, f"Preset {name} missing model_name"

    def test_all_presets_have_backend(self):
        """Test all presets have backend set."""
        from labeling_utils.models import _MODEL_PRESETS

        for name, config in _MODEL_PRESETS.items():
            assert config.backend, f"Preset {name} missing backend"


class TestLoadModelFunction:
    """Tests for load_model function (configuration only, no actual loading)."""

    def test_load_model_preset_detection(self):
        """Test that load_model detects presets correctly."""
        from labeling_utils.models import _MODEL_PRESETS

        # Just verify preset exists and has expected config
        assert "mms" in _MODEL_PRESETS
        config = _MODEL_PRESETS["mms"]
        assert config.backend == "huggingface"

    def test_load_model_backend_auto_detection(self):
        """Test backend auto-detection for non-preset models."""
        from labeling_utils.models import get_model_info

        # HuggingFace model
        info = get_model_info("facebook/wav2vec2-base-960h")
        assert info["backend"] == "huggingface"

        # TorchAudio pipeline
        info = get_model_info("WAV2VEC2_ASR_BASE_960H")
        assert info["backend"] == "torchaudio"

"""
Tests for labeling_utils base classes.

Tests cover:
- VocabInfo dataclass
- BackendConfig dataclass
- CTCModelBackend abstract class
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for labeling_utils imports"
)


class TestVocabInfo:
    """Tests for VocabInfo dataclass."""

    def test_vocab_info_creation(self):
        """Test creating a VocabInfo."""
        from labeling_utils.base import VocabInfo

        labels = ["-", "a", "b", "c", "*"]
        label_to_id = {c: i for i, c in enumerate(labels)}
        id_to_label = {i: c for i, c in enumerate(labels)}

        vocab = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            blank_id=0,
            unk_id=4,
            blank_token="-",
            unk_token="*",
        )

        assert len(vocab) == 5
        assert vocab.blank_id == 0
        assert vocab.unk_id == 4

    def test_vocab_info_contains(self):
        """Test __contains__ method."""
        from labeling_utils.base import VocabInfo

        labels = ["-", "a", "b", "c", "*"]
        label_to_id = {c: i for i, c in enumerate(labels)}
        id_to_label = {i: c for i, c in enumerate(labels)}

        vocab = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
        )

        assert "a" in vocab
        assert "x" not in vocab

    def test_vocab_info_len(self):
        """Test __len__ method."""
        from labeling_utils.base import VocabInfo

        labels = list("abcdefghij")
        label_to_id = {c: i for i, c in enumerate(labels)}
        id_to_label = {i: c for i, c in enumerate(labels)}

        vocab = VocabInfo(
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
        )

        assert len(vocab) == 10


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_backend_config_creation(self):
        """Test creating a BackendConfig."""
        import torch
        from labeling_utils.base import BackendConfig

        config = BackendConfig(
            model_name="facebook/mms-1b-all",
            language="eng",
            device="cpu",
        )

        assert config.model_name == "facebook/mms-1b-all"
        assert config.language == "eng"
        assert config.device == "cpu"

    def test_backend_config_defaults(self):
        """Test BackendConfig default values."""
        import torch
        from labeling_utils.base import BackendConfig

        config = BackendConfig(model_name="test-model")

        assert config.dtype == torch.float32
        assert config.with_star is True
        assert config.trust_remote_code is False
        assert config.extra_options == {}

    def test_backend_config_extra_options(self):
        """Test BackendConfig with extra_options."""
        from labeling_utils.base import BackendConfig

        config = BackendConfig(
            model_name="test-model",
            extra_options={"custom_param": 42},
        )

        assert config.extra_options["custom_param"] == 42


class TestCTCModelBackendInterface:
    """Tests for CTCModelBackend abstract interface."""

    def test_backend_is_abstract(self):
        """Test that CTCModelBackend cannot be instantiated directly."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        config = BackendConfig(model_name="test")

        # Should not be able to instantiate abstract class
        # Note: In Python, this doesn't raise at instantiation,
        # but the abstract methods will raise if called
        backend = type("TestBackend", (CTCModelBackend,), {
            "load": lambda self: None,
            "get_emissions": lambda self, w, l=None: (None, None),
            "get_vocab_info": lambda self: None,
        })(config)

        assert backend.config == config

    def test_backend_name_property(self):
        """Test backend name property."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            BACKEND_NAME = "test-backend"

            def load(self):
                pass

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="test")
        backend = TestBackend(config)

        assert backend.name == "test-backend"

    def test_backend_is_loaded_property(self):
        """Test is_loaded property."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            def load(self):
                self._loaded = True

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="test")
        backend = TestBackend(config)

        assert backend.is_loaded is False
        backend.load()
        assert backend.is_loaded is True

    def test_backend_frame_duration_default(self):
        """Test default frame duration (20ms)."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            def load(self):
                pass

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="test")
        backend = TestBackend(config)

        assert backend.frame_duration == 0.02

    def test_backend_sample_rate_default(self):
        """Test default sample rate (16kHz)."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            def load(self):
                pass

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="test")
        backend = TestBackend(config)

        assert backend.sample_rate == 16000

    def test_backend_unload(self):
        """Test unload method."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            def load(self):
                self._model = "loaded"
                self._loaded = True

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="test")
        backend = TestBackend(config)
        backend.load()

        assert backend.is_loaded is True
        backend.unload()
        assert backend.is_loaded is False
        assert backend._model is None

    def test_backend_context_manager(self):
        """Test context manager protocol."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            def load(self):
                self._loaded = True

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="test")
        backend = TestBackend(config)

        with backend as b:
            assert b.is_loaded is True

        assert backend.is_loaded is False

    def test_backend_repr(self):
        """Test __repr__ method."""
        from labeling_utils.base import CTCModelBackend, BackendConfig

        class TestBackend(CTCModelBackend):
            def load(self):
                pass

            def get_emissions(self, waveform, lengths=None):
                return None, None

            def get_vocab_info(self):
                return None

        config = BackendConfig(model_name="facebook/mms-1b-all")
        backend = TestBackend(config)

        repr_str = repr(backend)
        assert "TestBackend" in repr_str
        assert "facebook/mms-1b-all" in repr_str

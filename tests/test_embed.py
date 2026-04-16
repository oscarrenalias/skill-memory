"""Tests for memory.py _embed() function (bead B-22336fc3).

Coverage:
  (1) Single text → output shape (1, 384)
  (2) Batch of 2 texts → output shape (2, 384)
  (3) Output is L2-normalised — norm ≈ 1.0 per row (within 1e-5)
  (4) Output dtype is float32
  (5) Calling _embed twice reuses _ort_session singleton (identity check)
  (6) Calling _embed twice reuses _embed_tokenizer singleton (identity check)
  (7) token_type_ids missing from model inputs → RuntimeError is raised (known limitation)

Tests that call _embed() directly are gated with @_requires_numpy because the
function itself does `import numpy as np` internally.  All tests run cleanly
with just the managed venv's Python; they also pass on any interpreter that has
numpy installed.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

MEMORY_PY = Path(__file__).parent.parent / ".apm/skills/skill-memory/memory.py"
_VENV_PATH = str(MEMORY_PY.parent / ".venv")

try:
    import numpy as _np
    _numpy_available = True
except ImportError:
    _np = None
    _numpy_available = False

_requires_numpy = unittest.skipUnless(
    _numpy_available,
    "numpy not available — install numpy or set up the managed venv",
)


# ---------------------------------------------------------------------------
# Module loader helper
# ---------------------------------------------------------------------------

def _load_memory_module():
    """Load memory.py as a fresh module instance with _bootstrap() stubbed out.

    Each call returns a completely fresh module so singleton state does not
    leak between test cases.
    """
    unique_name = f"memory_embed_test_{id(object())}"
    with mock.patch("sys.prefix", _VENV_PATH):
        spec = importlib.util.spec_from_file_location(unique_name, str(MEMORY_PY))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(unique_name, None)
    # Ensure cold-start state
    mod._ort_session = None
    mod._embed_tokenizer = None
    return mod


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _FakeEncoding:
    """Minimal tokenizer encoding stub."""

    def __init__(self, seq_len: int = 10):
        self.ids = list(range(seq_len))
        self.attention_mask = [1] * seq_len
        self.type_ids = [0] * seq_len


class _FakeTokenizer:
    """Returns a _FakeEncoding for each text in encode_batch."""

    def encode_batch(self, texts):
        return [_FakeEncoding() for _ in texts]


class _FakeSession:
    """ONNX InferenceSession stub.

    Returns a last_hidden_state where CLS token (index 0) is a unit vector
    along dim 0, so _embed's L2-normalisation produces norm exactly 1.0.
    """

    def __init__(self, seq_len: int = 10):
        self._seq_len = seq_len

    def run(self, output_names, inputs):
        import numpy as np
        n = inputs["input_ids"].shape[0]
        state = np.zeros((n, self._seq_len, 384), dtype=np.float32)
        state[:, 0, 0] = 1.0  # CLS = [1, 0, …, 0] → already unit-norm
        return [state]


class _TokenTypeIdsRejectingSession:
    """Simulates an ONNX model that does not expose a token_type_ids input node.

    Documents the known limitation from the developer handoff: if the
    downloaded model variant omits token_type_ids from its named inputs,
    session.run() will raise RuntimeError.  A corrective bead should probe
    session.get_inputs() and omit token_type_ids when absent.
    """

    def run(self, output_names, inputs):
        if "token_type_ids" in inputs:
            raise RuntimeError(
                "Invalid Feed Input Name:token_type_ids. "
                "The model does not have an input of that name."
            )
        import numpy as np
        n = inputs["input_ids"].shape[0]
        state = np.zeros((n, 10, 384), dtype=np.float32)
        state[:, 0, 0] = 1.0
        return [state]


def _wire_stubs(mod, session=None, tokenizer=None):
    """Inject fake singletons directly into a freshly loaded module."""
    mod._ort_session = session if session is not None else _FakeSession()
    mod._embed_tokenizer = tokenizer if tokenizer is not None else _FakeTokenizer()


# ---------------------------------------------------------------------------
# (1) Single text → shape (1, 384)
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedSingleShape(unittest.TestCase):
    def test_single_text_shape_is_1_384(self):
        mod = _load_memory_module()
        _wire_stubs(mod)

        result = mod._embed(["hello world"])

        self.assertEqual(
            result.shape, (1, 384),
            f"Expected shape (1, 384) for single input; got {result.shape}",
        )


# ---------------------------------------------------------------------------
# (2) Batch of 2 texts → shape (2, 384)
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedBatchShape(unittest.TestCase):
    def test_batch_of_two_shape_is_2_384(self):
        mod = _load_memory_module()
        _wire_stubs(mod)

        result = mod._embed(["first sentence", "second sentence"])

        self.assertEqual(
            result.shape, (2, 384),
            f"Expected shape (2, 384) for batch of 2; got {result.shape}",
        )

    def test_batch_has_correct_row_and_dim_counts(self):
        mod = _load_memory_module()
        _wire_stubs(mod)

        result = mod._embed(["alpha", "beta"])

        self.assertEqual(result.shape[0], 2, "Expected 2 rows for 2-text batch")
        self.assertEqual(result.shape[1], 384, "Expected 384 embedding dimensions")


# ---------------------------------------------------------------------------
# (3) L2 normalisation — norm ≈ 1.0 per row (within 1e-5)
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedL2Norm(unittest.TestCase):
    def _check_norms(self, result):
        import numpy as np
        norms = np.linalg.norm(result, axis=1)
        for i, norm in enumerate(norms):
            self.assertAlmostEqual(
                float(norm), 1.0, delta=1e-5,
                msg=f"Row {i}: expected L2 norm ≈ 1.0, got {norm}",
            )

    def test_single_row_norm_is_one(self):
        mod = _load_memory_module()
        _wire_stubs(mod)
        result = mod._embed(["normalisation test"])
        self._check_norms(result)

    def test_batch_all_rows_norm_is_one(self):
        mod = _load_memory_module()
        _wire_stubs(mod)
        result = mod._embed(["first", "second"])
        self._check_norms(result)


# ---------------------------------------------------------------------------
# (4) Output dtype is float32
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedDtype(unittest.TestCase):
    def test_output_dtype_is_float32(self):
        import numpy as np
        mod = _load_memory_module()
        _wire_stubs(mod)

        result = mod._embed(["dtype check"])

        self.assertEqual(
            result.dtype, np.float32,
            f"Expected float32 output; got {result.dtype}",
        )


# ---------------------------------------------------------------------------
# (5) Session singleton: identity preserved across two _embed calls
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedSessionSingleton(unittest.TestCase):
    """_ort_session must not be replaced between successive _embed calls."""

    def test_session_identity_preserved_after_two_calls(self):
        mod = _load_memory_module()
        fake_session = _FakeSession()
        _wire_stubs(mod, session=fake_session)

        mod._embed(["first call"])
        session_after_first = mod._ort_session

        mod._embed(["second call"])
        session_after_second = mod._ort_session

        self.assertIs(
            session_after_first, session_after_second,
            "Session object was replaced between two _embed calls",
        )
        self.assertIs(
            session_after_first, fake_session,
            "Session identity no longer matches the pre-populated singleton",
        )

    def test_session_not_none_after_embed(self):
        mod = _load_memory_module()
        _wire_stubs(mod)
        mod._embed(["hello"])
        self.assertIsNotNone(
            mod._ort_session,
            "_ort_session was None after _embed completed",
        )


# ---------------------------------------------------------------------------
# (6) Tokenizer singleton: identity preserved across two _embed calls
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedTokenizerSingleton(unittest.TestCase):
    """_embed_tokenizer must not be replaced between successive _embed calls."""

    def test_tokenizer_identity_preserved_after_two_calls(self):
        mod = _load_memory_module()
        fake_tokenizer = _FakeTokenizer()
        _wire_stubs(mod, tokenizer=fake_tokenizer)

        mod._embed(["first"])
        tok_after_first = mod._embed_tokenizer

        mod._embed(["second"])
        tok_after_second = mod._embed_tokenizer

        self.assertIs(
            tok_after_first, tok_after_second,
            "Tokenizer object was replaced between two _embed calls",
        )
        self.assertIs(
            tok_after_first, fake_tokenizer,
            "Tokenizer identity no longer matches the pre-populated singleton",
        )

    def test_tokenizer_not_none_after_embed(self):
        mod = _load_memory_module()
        _wire_stubs(mod)
        mod._embed(["hello"])
        self.assertIsNotNone(
            mod._embed_tokenizer,
            "_embed_tokenizer was None after _embed completed",
        )


# ---------------------------------------------------------------------------
# (7) Known limitation: token_type_ids missing from model raises RuntimeError
# ---------------------------------------------------------------------------

@_requires_numpy
class TestEmbedTokenTypeIdsMissingLimitation(unittest.TestCase):
    """Documents the known limitation: models without token_type_ids input fail.

    If the downloaded bge-small-en-v1.5 ONNX variant does not expose a
    token_type_ids input node, _embed will raise RuntimeError.  A corrective
    bead should make the token_type_ids feed optional by probing
    session.get_inputs() and omitting the key when absent.
    """

    def test_missing_token_type_ids_input_raises(self):
        mod = _load_memory_module()
        _wire_stubs(mod, session=_TokenTypeIdsRejectingSession())

        with self.assertRaises(RuntimeError) as cm:
            mod._embed(["test text"])

        self.assertIn(
            "token_type_ids", str(cm.exception),
            "Expected RuntimeError mentioning token_type_ids",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for B-2760e5db: Download and commit tokenizer assets.

Validates that all required tokenizer asset files are present, well-formed,
and contain the expected structure for a BERT-compatible tokenizer.
"""
import json
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"

REQUIRED_FILES = [
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]

EXPECTED_SPECIAL_TOKENS = {"cls_token", "mask_token", "pad_token", "sep_token", "unk_token"}
EXPECTED_SPECIAL_TOKEN_VALUES = {
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "unk_token": "[UNK]",
}


def test_all_asset_files_exist():
    for filename in REQUIRED_FILES:
        path = ASSETS_DIR / filename
        assert path.exists(), f"Missing asset file: {filename}"
        assert path.stat().st_size > 0, f"Asset file is empty: {filename}"


def test_special_tokens_map_valid_json():
    path = ASSETS_DIR / "special_tokens_map.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict), "special_tokens_map.json must be a JSON object"


def test_special_tokens_map_has_required_keys():
    data = json.loads((ASSETS_DIR / "special_tokens_map.json").read_text())
    for key in EXPECTED_SPECIAL_TOKENS:
        assert key in data, f"special_tokens_map.json missing key: {key}"


def test_special_tokens_map_values():
    data = json.loads((ASSETS_DIR / "special_tokens_map.json").read_text())
    for key, expected_value in EXPECTED_SPECIAL_TOKEN_VALUES.items():
        actual = data.get(key)
        # value may be a string or a dict with "content" key
        if isinstance(actual, dict):
            actual = actual.get("content", actual)
        assert actual == expected_value, (
            f"special_tokens_map.json[{key!r}] = {actual!r}, expected {expected_value!r}"
        )


def test_tokenizer_config_valid_json():
    path = ASSETS_DIR / "tokenizer_config.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict), "tokenizer_config.json must be a JSON object"


def test_tokenizer_config_is_bert():
    data = json.loads((ASSETS_DIR / "tokenizer_config.json").read_text())
    tokenizer_class = data.get("tokenizer_class", "")
    assert "Bert" in tokenizer_class, (
        f"Expected a BERT tokenizer class, got {tokenizer_class!r}"
    )


def test_tokenizer_config_model_max_length():
    data = json.loads((ASSETS_DIR / "tokenizer_config.json").read_text())
    assert "model_max_length" in data, "tokenizer_config.json missing model_max_length"
    assert data["model_max_length"] == 512


def test_tokenizer_json_valid():
    path = ASSETS_DIR / "tokenizer.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict), "tokenizer.json must be a JSON object"
    assert "model" in data, "tokenizer.json missing 'model' key"
    assert data["model"].get("type") == "WordPiece", (
        f"Expected WordPiece model, got {data['model'].get('type')!r}"
    )


def test_tokenizer_json_has_vocab():
    data = json.loads((ASSETS_DIR / "tokenizer.json").read_text())
    vocab = data.get("model", {}).get("vocab", {})
    assert len(vocab) > 0, "tokenizer.json model vocab is empty"
    assert "[PAD]" in vocab, "vocab missing [PAD] token"
    assert "[UNK]" in vocab, "vocab missing [UNK] token"
    assert "[CLS]" in vocab, "vocab missing [CLS] token"
    assert "[SEP]" in vocab, "vocab missing [SEP] token"
    assert "[MASK]" in vocab, "vocab missing [MASK] token"


def test_vocab_txt_non_empty():
    path = ASSETS_DIR / "vocab.txt"
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) > 100, f"vocab.txt suspiciously short: {len(lines)} lines"


def test_vocab_txt_has_special_tokens():
    path = ASSETS_DIR / "vocab.txt"
    tokens = {l.strip() for l in path.read_text().splitlines() if l.strip()}
    for special in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        assert special in tokens, f"vocab.txt missing special token: {special}"

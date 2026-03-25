"""Load tokenizer from tokenizer.json via the tokenizers library."""

from pathlib import Path
from tokenizers import Tokenizer


def load_tokenizer(model_path: str | Path) -> Tokenizer:
    """Load a tokenizer from a model directory containing tokenizer.json."""
    model_path = Path(model_path)
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
    return Tokenizer.from_file(str(tokenizer_path))

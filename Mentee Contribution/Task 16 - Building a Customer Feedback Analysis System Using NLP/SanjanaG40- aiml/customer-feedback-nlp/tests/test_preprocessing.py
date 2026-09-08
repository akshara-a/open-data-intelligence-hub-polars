"""
Unit tests for the text preprocessing module.

Run with:  python -m pytest tests/test_preprocessing.py
or:        python tests/test_preprocessing.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    clean_text,
    preprocess_text,
    preprocess_dataframe,
    remove_stop_words,
    tokenize_text,
    NEGATION_WORDS,
)


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO World") == "hello world"

    def test_removes_urls(self):
        assert clean_text("Visit https://example.com now") == "visit now"

    def test_removes_emails(self):
        assert clean_text("Contact support@example.com please") == "contact please"

    def test_removes_handles(self):
        assert clean_text("Hey @john what's up") == "hey what's up"

    def test_removes_special_chars(self):
        # Note: basic punctuation (!, ., ,, ?) is preserved intentionally
        # because it can carry meaning; only other symbols are removed.
        assert clean_text("Great!!! ***app***") == "great!!! app"

    def test_collapses_whitespace(self):
        assert clean_text("Too   many     spaces") == "too many spaces"

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_empty_input(self):
        assert clean_text("") == ""


class TestRemoveStopWords:
    def test_keeps_negation(self):
        text = "this is not a good product"
        cleaned = remove_stop_words(text)
        # "not" must survive
        assert "not" in cleaned
        # common stop words removed
        assert "is" not in cleaned.split()

    def test_removes_stop_words(self):
        text = "the quick brown fox"
        cleaned = remove_stop_words(text)
        # "the" is a stop word
        assert "the" not in cleaned.split()
        # content words remain
        assert "quick" in cleaned


class TestPreprocessText:
    def test_basic_pipeline(self):
        result = preprocess_text(
            "The app is VERY slow!!! #help",
            clean=True,
            remove_stops=False,
            lemmatize=False,
        )
        assert result == "the app is very slow!!! help"

    def test_clean_false(self):
        result = preprocess_text(
            "The app is VERY slow!!!",
            clean=False,
        )
        assert result == "The app is VERY slow!!!"


class TestPreprocessDataFrame:
    def test_standard(self):
        df = pd.DataFrame({"text": ["Hello World", "Payment failed"]})
        result = preprocess_dataframe(df, text_column="text", new_column="clean")
        assert result["clean"].tolist() == ["hello world", "payment failed"]
        assert "text" in result.columns  # original preserved

    def test_overwrite(self):
        df = pd.DataFrame({"text": ["Hello World"]})
        result = preprocess_dataframe(df, text_column="text")
        assert result["text"].tolist() == ["hello world"]


class TestTokenize:
    def test_basic(self):
        tokens = tokenize_text("payment failed during checkout")
        assert "payment" in tokens
        assert "checkout" in tokens


if __name__ == "__main__":
    import traceback

    tests = [
        TestCleanText,
        TestRemoveStopWords,
        TestPreprocessText,
        TestPreprocessDataFrame,
        TestTokenize,
    ]
    passed = failed = 0
    for test_cls in tests:
        instance = test_cls()
        for name in dir(test_cls):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    print(f"  PASS  {test_cls.__name__}.{name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL  {test_cls.__name__}.{name}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

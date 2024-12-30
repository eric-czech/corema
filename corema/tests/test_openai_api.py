"""Tests for OpenAI API utilities."""

from corema.utils.openai_api import chunk_text
from corema.utils.openai_models import DEFAULT_MODEL
import tiktoken


def test_chunk_text_small_input() -> None:
    """Test that small inputs are returned as a single chunk."""
    text = "This is a small text"
    model = DEFAULT_MODEL
    chunks = list(chunk_text(text, model, max_chunk_tokens=100))
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_exact_size() -> None:
    """Test that text exactly matching chunk size is returned as one chunk."""
    # "This", "is", "a", "test" are typically 1 token each
    text = "This is a test"
    model = DEFAULT_MODEL
    chunks = list(chunk_text(text, model, max_chunk_tokens=4))
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_chunks() -> None:
    """Test that large text is split into multiple chunks with overlap."""
    # Create text that will definitely be multiple chunks
    text = "word " * 100  # Each "word " is typically 1-2 tokens
    model = DEFAULT_MODEL
    max_chunk_tokens = 20
    overlap_tokens = 5

    chunks = list(chunk_text(text, model, max_chunk_tokens, overlap_tokens))

    assert len(chunks) > 1  # Should be multiple chunks

    # Check that chunks have content
    for chunk in chunks:
        assert chunk.strip()  # Not empty
        assert "word" in chunk  # Contains expected content


def test_chunk_text_with_overlap() -> None:
    """Test that chunks overlap by the specified number of tokens."""
    # Use a simple repeating pattern where we know token boundaries
    text = "one two three four five six seven eight nine ten"
    model = DEFAULT_MODEL
    max_chunk_tokens = 4
    overlap_tokens = 2

    chunks = list(chunk_text(text, model, max_chunk_tokens, overlap_tokens))

    # We expect chunks like:
    # "one two three four"
    # "three four five six"
    # "five six seven eight"
    # "seven eight nine ten"

    assert len(chunks) > 1

    # Check overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        # The last words of one chunk should be the first words of the next
        words_current = chunks[i].split()
        words_next = chunks[i + 1].split()

        # Check that some words overlap
        assert any(word in words_next for word in words_current[-2:])


def test_chunk_text_empty_input() -> None:
    """Test that empty input returns empty iterator."""
    chunks = list(chunk_text("", DEFAULT_MODEL, max_chunk_tokens=10))
    assert len(chunks) == 0


def test_chunk_text_single_large_token() -> None:
    """Test handling of text with max_chunk_tokens=1."""
    # Use a string that will be split into multiple tokens
    text = "supercalifragilisticexpialidocious"
    model = DEFAULT_MODEL
    max_chunk_tokens = 1

    chunks = list(chunk_text(text, model, max_chunk_tokens=max_chunk_tokens))

    # Each chunk should be non-empty
    assert all(chunk.strip() for chunk in chunks)

    # Verify each chunk is exactly one token
    encoding = tiktoken.encoding_for_model(model)
    for chunk in chunks:
        assert (
            len(encoding.encode(chunk)) == 1
        ), f"Chunk '{chunk}' contains more than one token"

    # Verify concatenating chunks gives us back the original text
    assert "".join(chunks) == text

"""Unit tests for request validation."""

import pytest
from pydantic import ValidationError
from src.models import EmbeddingRequest, CompletionRequest


class TestEmbeddingRequestValidation:
    """Test validation for embedding requests."""

    def test_valid_single_text(self):
        """Test valid single text input."""
        request = EmbeddingRequest(input="hello world", model="test-model")
        assert request.input == "hello world"
        assert request.model == "test-model"

    def test_valid_batch_text(self):
        """Test valid batch text input."""
        texts = ["text1", "text2", "text3"]
        request = EmbeddingRequest(input=texts, model="test-model")
        assert request.input == texts

    def test_invalid_empty_string(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="", model="test-model")

    def test_invalid_empty_list(self):
        """Test that empty list is rejected."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input=[], model="test-model")

    def test_invalid_list_with_empty_strings(self):
        """Test that list with empty strings is rejected."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input=["valid", "", "also valid"], model="test-model")

    def test_missing_model(self):
        """Test that missing model is rejected."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="test")

    def test_encoding_format_default(self):
        """Test encoding_format defaults to 'float'."""
        request = EmbeddingRequest(input="test", model="test-model")
        assert request.encoding_format == "float"


class TestCompletionRequestValidation:
    """Test validation for completion requests."""

    def test_valid_simple_request(self):
        """Test valid simple completion request."""
        request = CompletionRequest(prompt="Hello", model="test-model")
        assert request.prompt == "Hello"
        assert request.model == "test-model"

    def test_valid_with_parameters(self):
        """Test valid request with all parameters."""
        request = CompletionRequest(
            prompt="Hello",
            model="test-model",
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
            n=2,
            stop=["\n", "."]
        )
        assert request.max_tokens == 100
        assert request.temperature == 0.8
        assert request.top_p == 0.9
        assert request.n == 2
        assert request.stop == ["\n", "."]

    def test_invalid_empty_prompt(self):
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="", model="test-model")

    def test_invalid_temperature_too_high(self):
        """Test that temperature > 2.0 is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", temperature=3.0)

    def test_invalid_temperature_negative(self):
        """Test that negative temperature is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", temperature=-0.1)

    def test_invalid_top_p_too_high(self):
        """Test that top_p > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", top_p=1.5)

    def test_invalid_top_p_negative(self):
        """Test that negative top_p is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", top_p=-0.1)

    def test_invalid_max_tokens_zero(self):
        """Test that max_tokens must be >= 1."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", max_tokens=0)

    def test_invalid_n_too_high(self):
        """Test that n > 10 is rejected."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", n=11)

    def test_invalid_n_zero(self):
        """Test that n must be >= 1."""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", model="test-model", n=0)

    def test_default_values(self):
        """Test that default values are set correctly."""
        request = CompletionRequest(prompt="test", model="test-model")
        assert request.max_tokens == 16
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.n == 1
        assert request.stream is False
        assert request.stop is None
        assert request.presence_penalty == 0.0
        assert request.frequency_penalty == 0.0

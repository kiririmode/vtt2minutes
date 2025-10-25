"""Tests for Amazon Bedrock integration."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from vtt2minutes.bedrock import BedrockError, BedrockMeetingMinutesGenerator


@pytest.fixture
def bedrock_generator() -> BedrockMeetingMinutesGenerator:
    """Shared fixture for BedrockMeetingMinutesGenerator setup."""
    return BedrockMeetingMinutesGenerator(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
    )


@pytest.fixture
def bedrock_generator_with_inference_profile() -> BedrockMeetingMinutesGenerator:
    """Shared fixture for BedrockMeetingMinutesGenerator with inference profile."""
    return BedrockMeetingMinutesGenerator(
        inference_profile_id="anthropic.claude-3-sonnet-20240229-v1:0",
    )


class TestBedrockMeetingMinutesGenerator:
    """Test cases for BedrockMeetingMinutesGenerator."""

    def _create_generator(self, model_id: str) -> BedrockMeetingMinutesGenerator:
        """Helper to create BedrockMeetingMinutesGenerator instance."""
        return BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id=model_id,
        )

    def _create_mock_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Helper to create mock response with given data."""
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(response_data).encode()
        return mock_response

    def _setup_boto3_client_mock(self, mock_boto3: Mock) -> tuple[Mock, Mock]:
        """Helper to setup boto3 client mocking with standard configuration.

        Returns:
            tuple[Mock, Mock]: (mock_runtime_client, mock_bedrock_client)
        """
        mock_runtime_client = Mock()
        mock_bedrock_client = Mock()

        def client_side_effect(service_name: str, **_kwargs: Any) -> Mock:
            if service_name == "bedrock-runtime":
                return mock_runtime_client
            elif service_name == "bedrock":
                return mock_bedrock_client
            else:
                raise ValueError(f"Unknown service: {service_name}")

        mock_boto3.client.side_effect = client_side_effect
        return mock_runtime_client, mock_bedrock_client

    def _assert_api_error(
        self,
        generator: BedrockMeetingMinutesGenerator,
        operation: str,
        expected_message: str,
        *args: Any,
    ) -> None:
        """Helper to assert API error behavior for any generator operation.

        Args:
            generator: The BedrockMeetingMinutesGenerator instance
            operation: The operation name to call
                (e.g., 'generate_minutes_from_markdown')
            expected_message: Expected error message pattern
            *args: Arguments to pass to the operation
        """
        operation_method = getattr(generator, operation)
        with pytest.raises(BedrockError, match=expected_message):
            operation_method(*args)

    def _setup_api_error_test(
        self,
        mock_boto3: Mock,
        service_method: str,
        error_code: str,
        operation_name: str,
    ) -> BedrockMeetingMinutesGenerator:
        """Helper to setup API error test with common pattern.

        Args:
            mock_boto3: Mocked boto3
            service_method: Method to mock
                (e.g., 'invoke_model', 'list_foundation_models')
            error_code: AWS error code (e.g., 'ValidationException')
            operation_name: AWS operation name for error (e.g., 'InvokeModel')

        Returns:
            BedrockMeetingMinutesGenerator: Configured generator instance
        """
        if service_method == "invoke_model":
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            error = ClientError({"Error": {"Code": error_code}}, operation_name)
            mock_client.invoke_model.side_effect = error
        else:
            _, mock_bedrock_client = self._setup_boto3_client_mock(mock_boto3)
            error = ClientError({"Error": {"Code": error_code}}, operation_name)
            getattr(mock_bedrock_client, service_method).side_effect = error

        return BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

    def test_init_with_credentials(self) -> None:
        """Test initialization with provided credentials."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-west-2",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        )

        assert generator.aws_access_key_id == "test_key"
        assert generator.aws_secret_access_key == "test_secret"
        assert generator.region_name == "us-west-2"
        assert generator.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_init_with_default_credentials(self) -> None:
        """Test initialization using default credential chain."""
        generator = BedrockMeetingMinutesGenerator()

        # When no explicit credentials are provided, should use None for AWS default
        assert generator.aws_access_key_id is None
        assert generator.aws_secret_access_key is None
        assert generator.aws_session_token is None

    def test_init_with_partial_credentials_access_key_only(self) -> None:
        """Test initialization with only access key provided."""
        generator = BedrockMeetingMinutesGenerator(aws_access_key_id="test_key")

        # Should store provided access key but leave secret as None
        assert generator.aws_access_key_id == "test_key"
        assert generator.aws_secret_access_key is None

    def test_init_with_partial_credentials_secret_only(self) -> None:
        """Test initialization with only secret key provided."""
        generator = BedrockMeetingMinutesGenerator(aws_secret_access_key="test_secret")

        # Should store provided secret but leave access key as None
        assert generator.aws_access_key_id is None
        assert generator.aws_secret_access_key == "test_secret"

    def test_init_with_custom_template_file(self) -> None:
        """Test initialization with custom template file."""
        template_path = Path("/custom/template.txt")
        generator = BedrockMeetingMinutesGenerator(prompt_template_file=template_path)

        assert generator.prompt_template_file == template_path

    @patch("vtt2minutes.bedrock.boto3")
    def test_validate_bedrock_access_success(self, mock_boto3: Mock) -> None:
        """Test successful access validation."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.list_foundation_models.return_value = {}

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            validate_access=True,
        )

        # Should not raise any exception
        assert generator is not None

    @pytest.mark.parametrize(
        "error_code,region_name",
        [
            ("UnauthorizedOperation", None),
            ("InvalidRequestException", "invalid-region"),
        ],
    )
    @patch("vtt2minutes.bedrock.boto3")
    def test_validate_bedrock_access_error(
        self, mock_boto3: Mock, error_code: str, region_name: str | None
    ) -> None:
        """Test access validation with various error types."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        error = ClientError({"Error": {"Code": error_code}}, "ListFoundationModels")
        mock_client.list_foundation_models.side_effect = error

        kwargs = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "validate_access": True,
        }
        if region_name:
            kwargs["region_name"] = region_name

        with pytest.raises(ValueError, match="Failed to initialize Bedrock client"):
            BedrockMeetingMinutesGenerator(**kwargs)  # type: ignore[misc]

    def test_create_prompt_japanese(self) -> None:
        """Test Japanese prompt creation."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        prompt = generator._create_prompt(
            "# Test Content\nSpeaker: Hello", "Test Meeting"
        )

        assert "以下をセクションとする議事録を作成してください" in prompt
        assert "Test Meeting" in prompt
        assert "Test Content" in prompt

    @pytest.mark.parametrize(
        "title,expected_title",
        [
            (None, "会議議事録"),  # Default title case
            ("Custom Meeting Title", "Custom Meeting Title"),  # Custom title case
        ],
    )
    def test_create_chat_prompt_with_title(
        self, title: str | None, expected_title: str
    ) -> None:
        """Test chat prompt creation with default and custom titles."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        markdown_content = "# Test Content\nSpeaker: Hello"
        if title is None:
            prompt = generator.create_chat_prompt(markdown_content)
        else:
            prompt = generator.create_chat_prompt(markdown_content, title)

        assert "以下をセクションとする議事録を作成してください" in prompt
        assert expected_title in prompt
        assert "Test Content" in prompt

    def test_create_chat_prompt_with_custom_template(self, tmp_path: Path) -> None:
        """Test chat prompt creation with custom template file."""
        # Create a custom template file
        template_file = tmp_path / "custom_template.txt"
        template_content = (
            "カスタムテンプレート: {title}\n内容: {markdown_content}\n以上です。"
        )
        template_file.write_text(template_content, encoding="utf-8")

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            prompt_template_file=template_file,
        )

        markdown_content = "# Test Content\nSpeaker: Hello"
        prompt = generator.create_chat_prompt(markdown_content, "Test Meeting")

        assert "カスタムテンプレート: Test Meeting" in prompt
        assert "内容: # Test Content\nSpeaker: Hello" in prompt
        assert "以上です。" in prompt

    def test_create_chat_prompt_template_file_not_found(self, tmp_path: Path) -> None:
        """Test chat prompt creation when template file doesn't exist."""
        non_existent_file = tmp_path / "nonexistent.txt"

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            prompt_template_file=non_existent_file,
        )

        markdown_content = "# Test Content\nSpeaker: Hello"
        prompt = generator.create_chat_prompt(markdown_content, "Test Meeting")

        # Should fall back to default template
        assert "以下をセクションとする議事録を作成してください" in prompt
        assert "Test Meeting" in prompt
        assert "Test Content" in prompt

    def _setup_invoke_model_test(
        self, model_id: str, mock_response_data: dict[str, Any]
    ) -> tuple[Mock, BedrockMeetingMinutesGenerator]:
        """Helper method to set up common test components for model invocation tests."""
        with patch("vtt2minutes.bedrock.boto3") as mock_boto3:
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client

            generator = BedrockMeetingMinutesGenerator(
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                model_id=model_id,
            )

            mock_response = {"body": Mock()}
            mock_response["body"].read.return_value = json.dumps(
                mock_response_data
            ).encode()
            mock_client.invoke_model.return_value = mock_response

            return mock_client, generator

    @pytest.mark.parametrize(
        "model_id,mock_response_data,expected_body_keys",
        [
            (
                "anthropic.claude-3-haiku-20240307-v1:0",
                {"anthropic_version": "bedrock-2023-05-31"},
                ["anthropic_version", "messages"],
            ),
            (
                "amazon.titan-text-express-v1",
                {"results": [{"outputText": "test response"}]},
                ["inputText", "textGenerationConfig"],
            ),
        ],
    )
    def test_invoke_model(
        self,
        model_id: str,
        mock_response_data: dict[str, Any],
        expected_body_keys: list[str],
    ) -> None:
        """Test model invocation for different model types."""
        mock_client, generator = self._setup_invoke_model_test(
            model_id, mock_response_data
        )

        result = generator._invoke_model("test prompt")

        assert result == mock_client.invoke_model.return_value
        mock_client.invoke_model.assert_called_once()

        # Check that the request body contains expected format
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        for key in expected_body_keys:
            assert key in body

    @pytest.mark.parametrize(
        "model_id,response_body,expected_text",
        [
            (
                "anthropic.claude-3-haiku-20240307-v1:0",
                {"content": [{"text": "Generated meeting minutes"}]},
                "Generated meeting minutes",
            ),
            (
                "amazon.titan-text-express-v1",
                {"results": [{"outputText": "Generated meeting minutes"}]},
                "Generated meeting minutes",
            ),
        ],
    )
    def test_extract_response_text(
        self, model_id: str, response_body: dict[str, Any], expected_text: str
    ) -> None:
        """Test response text extraction for different model types."""
        generator = self._create_generator(model_id)
        mock_response = self._create_mock_response(response_body)

        result = generator._extract_response_text(mock_response)
        assert result == expected_text

    def test_extract_response_text_claude_no_content(self) -> None:
        """Test response text extraction for Claude with no content."""
        generator = self._create_generator("anthropic.claude-3-haiku-20240307-v1:0")
        mock_response = self._create_mock_response({})

        with pytest.raises(BedrockError, match="No content found in Claude response"):
            generator._extract_response_text(mock_response)

    def test_extract_response_text_invalid_json(self) -> None:
        """Test response text extraction with invalid JSON."""
        generator = self._create_generator("anthropic.claude-3-haiku-20240307-v1:0")

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = b"invalid json"

        with pytest.raises(BedrockError, match="Failed to parse Bedrock response"):
            generator._extract_response_text(mock_response)

    @patch("vtt2minutes.bedrock.boto3")
    def test_generate_minutes_from_markdown_success(self, mock_boto3: Mock) -> None:
        """Test successful meeting minutes generation."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "# Meeting Minutes\n\nGenerated content"}]}
        ).encode()
        mock_client.invoke_model.return_value = mock_response

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )

        result = generator.generate_minutes_from_markdown(
            "# Test Meeting\n\nSpeaker: Hello world",
            title="Test Meeting",
        )

        assert result == "# Meeting Minutes\n\nGenerated content"

    @patch("vtt2minutes.bedrock.boto3")
    def test_generate_minutes_from_markdown_api_error(self, mock_boto3: Mock) -> None:
        """Test meeting minutes generation with API error."""
        generator = self._setup_api_error_test(
            mock_boto3, "invoke_model", "ValidationException", "InvokeModel"
        )

        self._assert_api_error(
            generator,
            "generate_minutes_from_markdown",
            "Bedrock API call failed",
            "# Test Meeting\n\nSpeaker: Hello world",
        )

    @patch("vtt2minutes.bedrock.boto3")
    def test_get_available_models_success(
        self, mock_boto3: Mock, bedrock_generator: BedrockMeetingMinutesGenerator
    ) -> None:
        """Test successful model listing."""
        _, mock_bedrock_client = self._setup_boto3_client_mock(mock_boto3)

        mock_bedrock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                    "responseStreamingSupported": True,
                },
                {
                    "modelId": "amazon.titan-text-express-v1",
                    "responseStreamingSupported": False,
                },
                {
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "responseStreamingSupported": True,
                },
            ]
        }

        models = bedrock_generator.get_available_models()

        # Should only return models with streaming support
        expected_models = [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
        ]
        assert models == expected_models

    @patch("vtt2minutes.bedrock.boto3")
    def test_get_available_models_api_error(self, mock_boto3: Mock) -> None:
        """Test model listing with API error."""
        generator = self._setup_api_error_test(
            mock_boto3,
            "list_foundation_models",
            "AccessDeniedException",
            "ListFoundationModels",
        )

        self._assert_api_error(
            generator,
            "get_available_models",
            "Failed to list Bedrock models",
        )

    @patch("vtt2minutes.bedrock.boto3.client")
    def test_generate_minutes_with_inference_profile(self, mock_boto3: Mock) -> None:
        """Test generate_minutes using inference_profile_id."""
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        # Mock successful response - Claude format
        mock_response = self._create_mock_response(
            {"content": [{"text": "Generated minutes"}]}
        )

        mock_client.invoke_model.return_value = mock_response

        # Create generator after mocking
        generator = BedrockMeetingMinutesGenerator(
            inference_profile_id="anthropic.claude-3-sonnet-20240229-v1:0",
        )

        result = generator.generate_minutes_from_markdown(
            "Test content", "Test meeting"
        )

        assert result == "Generated minutes"

        # Verify the correct parameters were used
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"


class TestBedrockError:
    """Test cases for BedrockError exception."""

    def test_bedrock_error_creation(self) -> None:
        """Test BedrockError exception creation."""
        error = BedrockError("Test error message")
        assert str(error) == "Test error message"

    def test_bedrock_error_inheritance(self) -> None:
        """Test that BedrockError inherits from Exception."""
        error = BedrockError("Test error")
        assert isinstance(error, Exception)

    def test_init_with_both_model_and_inference_profile_error(self) -> None:
        """Test initialization fails when both parameters are provided."""
        with pytest.raises(
            ValueError, match="Cannot specify both model_id and inference_profile_id"
        ):
            BedrockMeetingMinutesGenerator(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                inference_profile_id="test-profile",
            )

    def test_get_credentials_with_session_token(self) -> None:
        """Test that session token is included in credentials when provided."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_session_token="test_token",
        )

        credentials = generator._get_credential_dict()
        assert credentials["aws_session_token"] == "test_token"

    def test_init_with_inference_profile_id(self) -> None:
        """Test initialization with inference_profile_id."""
        generator = BedrockMeetingMinutesGenerator(
            inference_profile_id="test-profile",
        )

        assert generator.model_id is None
        assert generator.inference_profile_id == "test-profile"


class TestBedrockInferenceProfiles:
    """Test cases for inference profile support."""

    def _create_mock_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Helper to create mock response with given data."""
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(response_data).encode()
        return mock_response

    @patch("vtt2minutes.bedrock.boto3.client")
    def test_generate_minutes_with_claude_45_inference_profile(
        self, mock_boto3: Mock
    ) -> None:
        """Test that inference profile IDs with colons work correctly.

        Claude 4.5 inference profile IDs contain colons (e.g.,
        'us.anthropic.claude-sonnet-4-5-v2:0'). This test ensures that
        the updated boto3 version handles these IDs correctly.
        """
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        # Mock successful response
        mock_response = self._create_mock_response(
            {"content": [{"text": "Generated minutes"}]}
        )
        mock_client.invoke_model.return_value = mock_response

        # Create generator with Claude 4.5 style inference profile ID
        generator = BedrockMeetingMinutesGenerator(
            inference_profile_id="us.anthropic.claude-sonnet-4-5-v2:0",
        )

        result = generator.generate_minutes_from_markdown(
            "Test content", "Test meeting"
        )

        assert result == "Generated minutes"

        # Verify the correct parameters were used
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "us.anthropic.claude-sonnet-4-5-v2:0"

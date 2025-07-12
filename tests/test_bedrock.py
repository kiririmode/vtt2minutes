"""Tests for Amazon Bedrock integration."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from vtt2minutes.bedrock import BedrockError, BedrockMeetingMinutesGenerator


class TestBedrockMeetingMinutesGenerator:
    """Test cases for BedrockMeetingMinutesGenerator."""

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

    @patch("vtt2minutes.bedrock.boto3")
    def test_validate_bedrock_access_unauthorized(self, mock_boto3: Mock) -> None:
        """Test access validation with unauthorized error."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        error = ClientError(
            {"Error": {"Code": "UnauthorizedOperation"}}, "ListFoundationModels"
        )
        mock_client.list_foundation_models.side_effect = error

        with pytest.raises(ValueError, match="Failed to initialize Bedrock client"):
            BedrockMeetingMinutesGenerator(
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                validate_access=True,
            )

    @patch("vtt2minutes.bedrock.boto3")
    def test_validate_bedrock_access_invalid_region(self, mock_boto3: Mock) -> None:
        """Test access validation with invalid region error."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        error = ClientError(
            {"Error": {"Code": "InvalidRequestException"}}, "ListFoundationModels"
        )
        mock_client.list_foundation_models.side_effect = error

        with pytest.raises(ValueError, match="Failed to initialize Bedrock client"):
            BedrockMeetingMinutesGenerator(
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                region_name="invalid-region",
                validate_access=True,
            )

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

    def test_create_chat_prompt_default_title(self) -> None:
        """Test chat prompt creation with default title."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        markdown_content = "# Test Content\nSpeaker: Hello"
        prompt = generator.create_chat_prompt(markdown_content)

        assert "以下をセクションとする議事録を作成してください" in prompt
        assert "会議議事録" in prompt  # Default title
        assert "Test Content" in prompt

    def test_create_chat_prompt_custom_title(self) -> None:
        """Test chat prompt creation with custom title."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        markdown_content = "# Test Content\nSpeaker: Hello"
        prompt = generator.create_chat_prompt(markdown_content, "Custom Meeting Title")

        assert "以下をセクションとする議事録を作成してください" in prompt
        assert "Custom Meeting Title" in prompt
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

    @patch("vtt2minutes.bedrock.boto3")
    def test_invoke_model_claude(self, mock_boto3: Mock) -> None:
        """Test model invocation for Claude models."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(
            {"anthropic_version": "bedrock-2023-05-31"}
        ).encode()
        mock_client.invoke_model.return_value = mock_response

        result = generator._invoke_model("test prompt")

        assert result == mock_response
        mock_client.invoke_model.assert_called_once()

        # Check that the request body contains Claude-specific format
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert "anthropic_version" in body
        assert "messages" in body

    @patch("vtt2minutes.bedrock.boto3")
    def test_invoke_model_non_claude(self, mock_boto3: Mock) -> None:
        """Test model invocation for non-Claude models."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id="amazon.titan-text-express-v1",
        )

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(
            {"results": [{"outputText": "test response"}]}
        ).encode()
        mock_client.invoke_model.return_value = mock_response

        result = generator._invoke_model("test prompt")

        assert result == mock_response
        mock_client.invoke_model.assert_called_once()

        # Check that the request body contains Titan-specific format
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert "inputText" in body
        assert "textGenerationConfig" in body

    def test_extract_response_text_claude(self) -> None:
        """Test response text extraction for Claude models."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "Generated meeting minutes"}]}
        ).encode()

        result = generator._extract_response_text(mock_response)
        assert result == "Generated meeting minutes"

    def test_extract_response_text_claude_no_content(self) -> None:
        """Test response text extraction for Claude with no content."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps({}).encode()

        with pytest.raises(BedrockError, match="No content found in Claude response"):
            generator._extract_response_text(mock_response)

    def test_extract_response_text_titan(self) -> None:
        """Test response text extraction for Titan models."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            model_id="amazon.titan-text-express-v1",
        )

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(
            {"results": [{"outputText": "Generated meeting minutes"}]}
        ).encode()

        result = generator._extract_response_text(mock_response)
        assert result == "Generated meeting minutes"

    def test_extract_response_text_invalid_json(self) -> None:
        """Test response text extraction with invalid JSON."""
        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

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
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        error = ClientError({"Error": {"Code": "ValidationException"}}, "InvokeModel")
        mock_client.invoke_model.side_effect = error

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        with pytest.raises(BedrockError, match="Bedrock API call failed"):
            generator.generate_minutes_from_markdown(
                "# Test Meeting\n\nSpeaker: Hello world"
            )

    @patch("vtt2minutes.bedrock.boto3")
    def test_get_available_models_success(self, mock_boto3: Mock) -> None:
        """Test successful model listing."""
        mock_runtime_client = Mock()
        mock_bedrock_client = Mock()

        def client_side_effect(service_name: str, **_kwargs: str) -> Mock:
            if service_name == "bedrock-runtime":
                return mock_runtime_client
            elif service_name == "bedrock":
                return mock_bedrock_client
            else:
                raise ValueError(f"Unknown service: {service_name}")

        mock_boto3.client.side_effect = client_side_effect

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

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        models = generator.get_available_models()

        # Should only return models with streaming support
        expected_models = [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
        ]
        assert models == expected_models

    @patch("vtt2minutes.bedrock.boto3")
    def test_get_available_models_api_error(self, mock_boto3: Mock) -> None:
        """Test model listing with API error."""
        mock_runtime_client = Mock()
        mock_bedrock_client = Mock()

        def client_side_effect(service_name: str, **_kwargs: str) -> Mock:
            if service_name == "bedrock-runtime":
                return mock_runtime_client
            elif service_name == "bedrock":
                return mock_bedrock_client
            else:
                raise ValueError(f"Unknown service: {service_name}")

        mock_boto3.client.side_effect = client_side_effect

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException"}}, "ListFoundationModels"
        )
        mock_bedrock_client.list_foundation_models.side_effect = error

        generator = BedrockMeetingMinutesGenerator(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        with pytest.raises(BedrockError, match="Failed to list Bedrock models"):
            generator.get_available_models()


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

    @patch("vtt2minutes.bedrock.boto3.client")
    def test_generate_minutes_with_inference_profile(self, mock_boto3: Mock) -> None:
        """Test generate_minutes using inference_profile_id."""
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        # Mock successful response - Claude format
        mock_response = {
            "body": Mock(),
        }
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "Generated minutes"}]}
        ).encode()

        mock_client.invoke_model.return_value = mock_response

        generator = BedrockMeetingMinutesGenerator(
            # Include "claude" in the ID for proper response parsing
            inference_profile_id="anthropic.claude-3-sonnet-20240229-v1:0",
        )

        result = generator.generate_minutes_from_markdown(
            "Test content", "Test meeting"
        )

        assert result == "Generated minutes"

        # Verify the correct parameters were used
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"

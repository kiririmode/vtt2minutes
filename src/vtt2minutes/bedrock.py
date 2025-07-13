"""Amazon Bedrock integration for AI-powered meeting minutes generation."""

import json
import sys
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import (  # type: ignore[import-untyped]
    BotoCoreError,
    ClientError,
)


class BedrockMeetingMinutesGenerator:
    """Generate meeting minutes using Amazon Bedrock."""

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str = "ap-northeast-1",
        model_id: str | None = None,
        inference_profile_id: str | None = None,
        validate_access: bool = False,
        prompt_template_file: str | Path | None = None,
    ) -> None:
        """Initialize the Bedrock client.

        Args:
            aws_access_key_id: AWS access key ID (optional, uses default
                credentials chain if not provided)
            aws_secret_access_key: AWS secret access key (optional, uses default
                credentials chain if not provided)
            aws_session_token: AWS session token (optional, uses default
                credentials chain if not provided)
            region_name: AWS region name
            model_id: Bedrock model ID to use
                (mutually exclusive with inference_profile_id)
            inference_profile_id: Bedrock inference profile ID to use
                (mutually exclusive with model_id)
            validate_access: Whether to validate access during initialization
            prompt_template_file: Path to custom prompt template file
        """
        self._setup_client_configuration(
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            region_name,
            model_id,
            inference_profile_id,
            prompt_template_file,
            validate_access,
        )

    def _setup_client_configuration(
        self,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        aws_session_token: str | None,
        region_name: str,
        model_id: str | None,
        inference_profile_id: str | None,
        prompt_template_file: str | Path | None,
        validate_access: bool,
    ) -> None:
        """Setup client configuration in structured steps.

        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            region_name: AWS region name
            model_id: Bedrock model ID
            inference_profile_id: Bedrock inference profile ID
            prompt_template_file: Path to custom prompt template file
            validate_access: Whether to validate access
        """
        # Validate input parameters
        self._validate_model_parameters(model_id, inference_profile_id)

        # Set instance attributes
        self._set_instance_attributes(
            model_id,
            inference_profile_id,
            region_name,
            prompt_template_file,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
        )

        # Initialize Bedrock client
        self._initialize_bedrock_client(validate_access)

    def _validate_model_parameters(
        self, model_id: str | None, inference_profile_id: str | None
    ) -> None:
        """Validate model_id and inference_profile_id parameters.

        Args:
            model_id: Bedrock model ID
            inference_profile_id: Bedrock inference profile ID

        Raises:
            ValueError: If both or neither parameters are provided
        """
        if model_id is not None and inference_profile_id is not None:
            raise ValueError(
                "Cannot specify both model_id and inference_profile_id. "
                "Please specify only one."
            )

    def _set_instance_attributes(
        self,
        model_id: str | None,
        inference_profile_id: str | None,
        region_name: str,
        prompt_template_file: str | Path | None,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        aws_session_token: str | None,
    ) -> None:
        """Set instance attributes with default values.

        Args:
            model_id: Bedrock model ID
            inference_profile_id: Bedrock inference profile ID
            region_name: AWS region name
            prompt_template_file: Path to custom prompt template file
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
        """
        # Set model configuration with default
        if model_id is None and inference_profile_id is None:
            model_id = "anthropic.claude-sonnet-4-20250514-v1:0"

        self.model_id = model_id
        self.inference_profile_id = inference_profile_id
        self.region_name = region_name
        self.prompt_template_file = (
            Path(prompt_template_file) if prompt_template_file else None
        )

        # Store credentials
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

    def _initialize_bedrock_client(self, validate_access: bool) -> None:
        """Initialize the Bedrock client and optionally validate access.

        Args:
            validate_access: Whether to validate access during initialization

        Raises:
            ValueError: If client initialization fails
        """
        try:
            client_kwargs = self._build_client_kwargs()
            self.bedrock_client = boto3.client("bedrock-runtime", **client_kwargs)  # type: ignore[assignment]

            if validate_access:
                self._validate_bedrock_access()
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {e}") from e

    def _build_client_kwargs(self) -> dict[str, str]:
        """Build client configuration kwargs.

        Returns:
            Dictionary of client configuration parameters
        """
        return self._create_aws_client_config()

    def _create_aws_client_config(self) -> dict[str, str]:
        """Create AWS client configuration dictionary.

        Returns:
            Dictionary with AWS client configuration
        """
        client_kwargs = {"region_name": self.region_name}

        if self._has_explicit_credentials():
            client_kwargs.update(self._get_credential_dict())

        return client_kwargs

    def _has_explicit_credentials(self) -> bool:
        """Check if explicit credentials are provided.

        Returns:
            True if credentials are explicitly provided
        """
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    def _get_credential_dict(self) -> dict[str, str]:
        """Get credential dictionary for client configuration.

        Returns:
            Dictionary with credential configuration
        """
        credentials: dict[str, str] = {
            "aws_access_key_id": self.aws_access_key_id or "",
            "aws_secret_access_key": self.aws_secret_access_key or "",
        }

        if self.aws_session_token:
            credentials["aws_session_token"] = self.aws_session_token

        return credentials

    def generate_minutes_from_markdown(
        self,
        markdown_content: str,
        title: str = "会議議事録",
    ) -> str:
        """Generate Japanese meeting minutes from preprocessed Markdown content.

        Args:
            markdown_content: Preprocessed transcript in Markdown format
            title: Title for the meeting minutes

        Returns:
            Generated meeting minutes in Markdown format

        Raises:
            BedrockError: If the API call fails
        """
        prompt = self._create_prompt(markdown_content, title)

        try:
            response = self._invoke_model(prompt)
            return self._extract_response_text(response)
        except (ClientError, BotoCoreError) as e:
            raise BedrockError(f"Bedrock API call failed: {e}") from e
        except Exception as e:
            raise BedrockError(
                f"Unexpected error during minutes generation: {e}"
            ) from e

    def create_chat_prompt(
        self, markdown_content: str, title: str = "会議議事録"
    ) -> str:
        """Create a chat prompt for external AI services like ChatGPT.

        Args:
            markdown_content: Preprocessed transcript in Markdown format
            title: Title for the meeting minutes

        Returns:
            Formatted prompt string for chat services

        Raises:
            BedrockError: If prompt creation fails
        """
        return self._create_prompt(markdown_content, title)

    def _validate_bedrock_access(self) -> None:
        """Validate that we can access Bedrock with the current credentials.

        Raises:
            BedrockError: If credentials or region are invalid
        """
        try:
            # Create client configuration for bedrock service
            client_kwargs = {"region_name": self.region_name}

            # Only add credentials if they are explicitly provided
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_kwargs.update(
                    {
                        "aws_access_key_id": self.aws_access_key_id,
                        "aws_secret_access_key": self.aws_secret_access_key,
                    }
                )
                if self.aws_session_token:
                    client_kwargs["aws_session_token"] = self.aws_session_token

            # Try to get foundation models to validate access
            bedrock_client = boto3.client("bedrock", **client_kwargs)  # type: ignore[assignment]
            bedrock_client.list_foundation_models()  # type: ignore[misc]
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")  # type: ignore[misc]
            if error_code == "UnauthorizedOperation":
                raise BedrockError(
                    "AWS credentials are invalid or lack Bedrock permissions. "
                    "Please check your AWS credentials configuration."
                ) from e
            elif error_code == "InvalidRequestException":
                raise BedrockError(
                    f"Bedrock is not available in region '{self.region_name}'. "
                    f"Please use a supported region."
                ) from e
            else:
                raise BedrockError(f"AWS API error: {error_code}") from e
        except BotoCoreError as e:
            raise BedrockError(f"AWS connection error: {e}") from e

    def _create_prompt(self, markdown_content: str, title: str) -> str:
        """Create a prompt for Japanese meeting minutes generation.

        Args:
            markdown_content: Preprocessed transcript content
            title: Meeting title

        Returns:
            Formatted prompt string
        """
        # Try custom template first
        if template_content := self._load_custom_template():
            return self._substitute_placeholders(
                template_content, markdown_content, title
            )

        # Try default template
        if template_content := self._load_default_template():
            return self._substitute_placeholders(
                template_content, markdown_content, title
            )

        # Use hardcoded fallback
        return self._create_fallback_prompt(markdown_content, title)

    def _load_custom_template(self) -> str | None:
        """Load custom template file if available.

        Returns:
            Template content or None if not available
        """
        if not (self.prompt_template_file and self.prompt_template_file.exists()):
            return None

        try:
            return self.prompt_template_file.read_text(encoding="utf-8")
        except Exception as e:
            raise BedrockError(
                f"Failed to read prompt template file {self.prompt_template_file}: {e}"
            ) from e

    def _load_default_template(self) -> str | None:
        """Load default template file.

        Returns:
            Template content or None if not available
        """
        default_path = self._get_default_template_path()
        if not default_path.exists():
            return None

        try:
            return default_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _get_default_template_path(self) -> Path:
        """Get path to default template file.

        Returns:
            Path to default template
        """
        if getattr(sys, "frozen", False):
            # Running in a PyInstaller bundle
            bundle_dir = Path(getattr(sys, "_MEIPASS", ""))
            return bundle_dir / "prompt_templates" / "default.txt"
        else:
            # Running in normal Python environment
            return (
                Path(__file__).parent.parent.parent / "prompt_templates" / "default.txt"
            )

    def _create_fallback_prompt(self, markdown_content: str, title: str) -> str:
        """Create hardcoded fallback prompt.

        Args:
            markdown_content: Preprocessed transcript content
            title: Meeting title

        Returns:
            Hardcoded prompt string
        """
        template = (
            "以下の前処理済み会議記録から、構造化された議事録を作成してください。\n\n"
            "要件:\n"
            "1. 議題、決定事項、アクションアイテムを明確に抽出する\n"
            "2. 発言者の意図を正確に反映する\n"
            "3. 時系列順に整理する\n"
            "4. 重要なポイントを強調する\n"
            "5. Markdown形式で出力する\n\n"
            "タイトル: {title}\n\n"
            "前処理済み会議記録:\n{markdown_content}\n\n"
            "以下の構造で議事録を作成してください:\n"
            "- # タイトル\n"
            "- ## 会議概要 (日時、参加者、目的)\n"
            "- ## 主要な議題\n"
            "- ## 決定事項\n"
            "- ## アクションアイテム\n"
            "- ## 次回までの課題\n"
            "- ## 詳細な議論内容\n\n"
            "議事録:"
        )
        return self._substitute_placeholders(template, markdown_content, title)

    def _substitute_placeholders(
        self, template: str, markdown_content: str, title: str
    ) -> str:
        """Substitute placeholders in the template with actual values.

        Args:
            template: Template string with placeholders
            markdown_content: Preprocessed transcript content
            title: Meeting title

        Returns:
            Template with placeholders substituted
        """
        placeholders = {
            "markdown_content": markdown_content,
            "title": title,
        }

        result = template
        for placeholder, value in placeholders.items():
            result = result.replace(f"{{{placeholder}}}", str(value))

        return result

    def _invoke_model(self, prompt: str) -> dict[str, Any]:
        """Invoke the Bedrock model with the given prompt.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Model response dictionary

        Raises:
            ClientError: If the API call fails
        """
        # Prepare the request body based on the model
        model_identifier = self.model_id or self.inference_profile_id
        if model_identifier and "claude" in model_identifier.lower():
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
            )
        else:
            # Default format for other models
            body = json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 4000,
                        "stopSequences": [],
                        "temperature": 0.3,
                        "topP": 0.9,
                    },
                }
            )

        # Use the appropriate identifier for the invoke_model call
        invoke_params = {
            "contentType": "application/json",
            "accept": "application/json",
            "body": body,
        }

        if self.model_id:
            invoke_params["modelId"] = self.model_id
        elif self.inference_profile_id:
            invoke_params["modelId"] = self.inference_profile_id

        response = self.bedrock_client.invoke_model(**invoke_params)  # type: ignore[misc]

        return response  # type: ignore[return-value]

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        """Extract text from the model response.

        Args:
            response: Bedrock model response

        Returns:
            Generated text content

        Raises:
            BedrockError: If response parsing fails
        """
        try:
            response_body = json.loads(response["body"].read())
            return self._parse_response_body(response_body)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise BedrockError(f"Failed to parse Bedrock response: {e}") from e

    def _parse_response_body(self, response_body: dict[str, Any]) -> str:
        """Parse the response body to extract text content.

        Args:
            response_body: Parsed JSON response body

        Returns:
            Extracted text content

        Raises:
            BedrockError: If response format is unknown
        """
        # Handle Claude models
        model_identifier = self.model_id or self.inference_profile_id
        if model_identifier and "claude" in model_identifier.lower():
            return self._extract_claude_response(response_body)

        # Handle other models (Titan, etc.)
        return self._extract_standard_response(response_body)

    def _extract_claude_response(self, response_body: dict[str, Any]) -> str:
        """Extract text from Claude model response."""
        if response_body.get("content"):
            return response_body["content"][0]["text"]
        else:
            raise BedrockError("No content found in Claude response")

    def _extract_standard_response(self, response_body: dict[str, Any]) -> str:
        """Extract text from standard model response."""
        if "results" in response_body:
            return response_body["results"][0]["outputText"]
        elif "outputText" in response_body:
            return response_body["outputText"]
        else:
            raise BedrockError(f"Unknown response format: {response_body}")

    def get_available_models(self) -> list[str]:
        """Get list of available Bedrock models.

        Returns:
            List of available model IDs

        Raises:
            BedrockError: If the API call fails
        """
        try:
            # Create client configuration for bedrock service
            client_kwargs = {"region_name": self.region_name}

            # Only add credentials if they are explicitly provided
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_kwargs.update(
                    {
                        "aws_access_key_id": self.aws_access_key_id,
                        "aws_secret_access_key": self.aws_secret_access_key,
                    }
                )
                if self.aws_session_token:
                    client_kwargs["aws_session_token"] = self.aws_session_token

            bedrock_client = boto3.client("bedrock", **client_kwargs)  # type: ignore[assignment]

            response: dict[str, Any] = bedrock_client.list_foundation_models()  # type: ignore[misc]
            model_summaries: list[dict[str, Any]] = response.get("modelSummaries", [])  # type: ignore[misc]
            return [
                str(model["modelId"])  # type: ignore[misc]
                for model in model_summaries  # type: ignore[misc]
                if bool(model.get("responseStreamingSupported", False))  # type: ignore[misc]
            ]

        except (ClientError, BotoCoreError) as e:
            raise BedrockError(f"Failed to list Bedrock models: {e}") from e


class BedrockError(Exception):
    """Custom exception for Bedrock-related errors."""

    pass

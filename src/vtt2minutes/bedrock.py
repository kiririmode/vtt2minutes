"""Amazon Bedrock integration for AI-powered meeting minutes generation."""

import json
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
        # Validate that exactly one of model_id or inference_profile_id is provided
        if model_id is not None and inference_profile_id is not None:
            raise ValueError(
                "Cannot specify both model_id and inference_profile_id. "
                "Please specify only one."
            )
        if model_id is None and inference_profile_id is None:
            # Default to Claude Sonnet 4 if neither is specified
            model_id = "anthropic.claude-sonnet-4-20250514-v1:0"

        self.model_id = model_id
        self.inference_profile_id = inference_profile_id
        self.region_name = region_name
        self.prompt_template_file = (
            Path(prompt_template_file) if prompt_template_file else None
        )

        # Store provided credentials (may be None to use default credential chain)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

        # Initialize Bedrock client
        try:
            # Create client configuration
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

            self.bedrock_client = boto3.client("bedrock-runtime", **client_kwargs)  # type: ignore[assignment]

            # Validate credentials and region by attempting to list models
            if validate_access:
                self._validate_bedrock_access()
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {e}") from e

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
        # If custom template file is provided, use it
        if self.prompt_template_file and self.prompt_template_file.exists():
            try:
                template_content = self.prompt_template_file.read_text(encoding="utf-8")
                return self._substitute_placeholders(
                    template_content, markdown_content, title
                )
            except Exception as e:
                raise BedrockError(
                    f"Failed to read prompt template file "
                    f"{self.prompt_template_file}: {e}"
                ) from e

        # Fallback to default prompts
        return self._get_default_prompt(markdown_content, title)

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

    def _get_default_prompt(self, markdown_content: str, title: str) -> str:
        """Get the default prompt template for Japanese meeting minutes.

        Args:
            markdown_content: Preprocessed transcript content
            title: Meeting title

        Returns:
            Default Japanese prompt string
        """
        return (
            f"以下の前処理済み会議記録から、構造化された議事録を作成してください。\n\n"
            f"要件:\n"
            f"1. 議題、決定事項、アクションアイテムを明確に抽出する\n"
            f"2. 発言者の意図を正確に反映する\n"
            f"3. 時系列順に整理する\n"
            f"4. 重要なポイントを強調する\n"
            f"5. Markdown形式で出力する\n\n"
            f"タイトル: {title}\n\n"
            f"前処理済み会議記録:\n{markdown_content}\n\n"
            f"以下の構造で議事録を作成してください:\n"
            f"- # タイトル\n"
            f"- ## 会議概要 (日時、参加者、目的)\n"
            f"- ## 主要な議題\n"
            f"- ## 決定事項\n"
            f"- ## アクションアイテム\n"
            f"- ## 次回までの課題\n"
            f"- ## 詳細な議論内容\n\n"
            f"議事録:"
        )

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

            # Handle Claude models
            model_identifier = self.model_id or self.inference_profile_id
            if model_identifier and "claude" in model_identifier.lower():
                if response_body.get("content"):
                    return response_body["content"][0]["text"]
                else:
                    raise BedrockError("No content found in Claude response")

            # Handle other models (Titan, etc.)
            elif "results" in response_body:
                return response_body["results"][0]["outputText"]
            elif "outputText" in response_body:
                return response_body["outputText"]
            else:
                raise BedrockError(f"Unknown response format: {response_body}")

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise BedrockError(f"Failed to parse Bedrock response: {e}") from e

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

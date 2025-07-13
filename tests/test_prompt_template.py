"""Tests for prompt template functionality."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import Mock, patch

import pytest

from vtt2minutes.bedrock import BedrockError, BedrockMeetingMinutesGenerator


class TestPromptTemplateFeature:
    """Test prompt template functionality."""

    def _create_temp_template_file(self, content: str) -> Path:
        """Helper to create temporary template file with given content."""
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def _test_template_file_usage(
        self, template_content: str, expected_result: str
    ) -> None:
        """Helper to test template file usage with given content and expected result."""
        template_path = self._create_temp_template_file(template_content)

        try:
            generator = BedrockMeetingMinutesGenerator(
                prompt_template_file=template_path
            )
            result = generator._create_prompt("Test content", "Test Title")
            assert expected_result in result or result == expected_result
        finally:
            template_path.unlink()

    def test_substitute_placeholders(self) -> None:
        """Test placeholder substitution."""
        generator = BedrockMeetingMinutesGenerator()
        template = "Title: {title}\nContent: {markdown_content}"

        result = generator._substitute_placeholders(
            template, "Test content", "Test Title"
        )

        expected = "Title: Test Title\nContent: Test content"
        assert result == expected

    def test_fallback_prompt_when_no_template(self) -> None:
        """Test fallback hardcoded prompt when no template files are available."""
        # Use a nonexistent custom template and ensure default template doesn't exist
        nonexistent_path = Path("/nonexistent/template.txt")

        generator = BedrockMeetingMinutesGenerator(
            prompt_template_file=nonexistent_path
        )

        # Mock the default template path to not exist
        import unittest.mock

        with unittest.mock.patch("pathlib.Path.exists", return_value=False):
            result = generator._create_prompt("Test content", "Test Title")

        assert "以下の前処理済み会議記録から" in result
        assert "Test Title" in result
        assert "Test content" in result

    def test_custom_template_file_usage(self) -> None:
        """Test using custom template file."""
        template_content = "Custom template: {title}\n{markdown_content}"
        expected = "Custom template: Test Title\nTest content"
        self._test_template_file_usage(template_content, expected)

    def test_nonexistent_template_file_fallback(self) -> None:
        """Test fallback to default when template file doesn't exist."""
        nonexistent_path = Path("/nonexistent/template.txt")

        generator = BedrockMeetingMinutesGenerator(
            prompt_template_file=nonexistent_path
        )

        result = generator._create_prompt("Test content", "Test Title")

        # Should fall back to default prompt
        assert "以下をセクションとする議事録を作成してください" in result

    def test_template_file_read_error(self) -> None:
        """Test error handling when template file can't be read."""
        # Create a file that we can't read by using a directory path
        directory_path = Path("/tmp")

        generator = BedrockMeetingMinutesGenerator(prompt_template_file=directory_path)

        with pytest.raises(BedrockError, match="Failed to read prompt template file"):
            generator._create_prompt("Test content", "Test Title")

    def test_template_with_all_placeholders(self) -> None:
        """Test template with all available placeholders."""
        template_content = (
            "Title: {title}\n"
            "Content: {markdown_content}\n"
            "Japanese meeting minutes instruction."
        )
        template_path = self._create_temp_template_file(template_content)

        try:
            generator = BedrockMeetingMinutesGenerator(
                prompt_template_file=template_path
            )
            result = generator._create_prompt(
                "Meeting transcript here", "Weekly Standup"
            )

            expected = (
                "Title: Weekly Standup\n"
                "Content: Meeting transcript here\n"
                "Japanese meeting minutes instruction."
            )
            assert result == expected
        finally:
            template_path.unlink()

    @patch("boto3.client")
    def test_integration_with_generate_minutes(self, mock_boto_client: Mock) -> None:
        """Test integration of template with generate_minutes_from_markdown."""
        # Mock Bedrock client
        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        # Mock response
        mock_response = {"body": Mock()}
        mock_response[
            "body"
        ].read.return_value = '{"content": [{"text": "Generated minutes"}]}'
        mock_client.invoke_model.return_value = mock_response

        template_content = "Generate minutes for: {title}\nFrom: {markdown_content}"

        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(template_content)
            template_path = Path(f.name)

        try:
            generator = BedrockMeetingMinutesGenerator(
                prompt_template_file=template_path
            )

            result = generator.generate_minutes_from_markdown(
                "Test transcript", "Test Meeting"
            )

            assert result == "Generated minutes"

            # Verify the custom prompt was used
            mock_client.invoke_model.assert_called_once()
            call_args = mock_client.invoke_model.call_args
            body_json = call_args[1]["body"]

            # The body should contain our custom template
            assert "Generate minutes for: Test Meeting" in body_json
            assert "From: Test transcript" in body_json

        finally:
            template_path.unlink()

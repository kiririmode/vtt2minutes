"""Tests for word replacement functionality."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from vtt2minutes.parser import VTTCue
from vtt2minutes.preprocessor import PreprocessingConfig, TextPreprocessor


class TestWordReplacement:
    """Test word replacement functionality."""

    def _create_temp_rules_file(self, content: str) -> Path:
        """Helper to create temporary rules file with given content."""
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def _test_rules_file_loading(
        self, content: str, expected_rules: dict[str, str]
    ) -> None:
        """Helper to test rules file loading with given content and expected result."""
        rules_file = self._create_temp_rules_file(content)

        try:
            config = PreprocessingConfig(replacement_rules_file=rules_file)
            assert config.replacement_rules == expected_rules
        finally:
            rules_file.unlink()

    def _test_word_replacement(
        self,
        rules: dict[str, str],
        input_text: str,
        expected_output: str,
        enabled: bool = True,
    ) -> None:
        """Helper to test word replacement with given rules and text."""
        config = PreprocessingConfig(
            replacement_rules=rules,
            enable_word_replacement=enabled,
        )
        preprocessor = TextPreprocessor(config)
        result = preprocessor._apply_word_replacement(input_text)
        assert result == expected_output

    def test_basic_replacement(self) -> None:
        """Test basic word replacement functionality."""
        rules = {"ベッドロック": "Bedrock", "エス3": "S3"}
        input_text = "今日はベッドロックとエス3について話します。"
        expected_output = "今日はBedrockとS3について話します。"
        self._test_word_replacement(rules, input_text, expected_output)

    def test_replacement_order(self) -> None:
        """Test that longer words are replaced before shorter ones."""
        rules = {
            "アマゾンウェブサービス": "Amazon Web Services",
            "アマゾン": "Amazon",
        }
        # Test that the longer phrase is replaced, not just "アマゾン"
        input_text = "アマゾンウェブサービスを使用します。"
        expected_output = "Amazon Web Servicesを使用します。"
        self._test_word_replacement(rules, input_text, expected_output)

    def test_no_replacement_when_disabled(self) -> None:
        """Test that replacement is skipped when disabled."""
        rules = {"ベッドロック": "Bedrock"}
        input_text = "ベッドロックについて"
        expected_output = "ベッドロックについて"  # Should remain unchanged
        self._test_word_replacement(rules, input_text, expected_output, enabled=False)

    def test_empty_replacement_rules(self) -> None:
        """Test behavior with empty replacement rules."""
        rules: dict[str, str] = {}  # Empty rules
        input_text = "ベッドロックについて話します。"
        expected_output = "ベッドロックについて話します。"  # Should remain unchanged
        self._test_word_replacement(rules, input_text, expected_output)

    def test_replacement_rules_file_loading(self) -> None:
        """Test loading replacement rules from file."""
        rules_content = """# Test replacement rules
ベッドロック -> Bedrock
エス3 -> S3
# Comment line
アマゾン -> Amazon

# Empty line above is ignored
"""
        expected_rules = {
            "ベッドロック": "Bedrock",
            "エス3": "S3",
            "アマゾン": "Amazon",
        }
        self._test_rules_file_loading(rules_content, expected_rules)

    def test_replacement_rules_file_invalid_format(self) -> None:
        """Test handling of invalid format in replacement rules file."""
        rules_content = """# Test replacement rules
ベッドロック -> Bedrock
invalid_line_without_arrow
エス3 -> S3
"""
        # Should load valid lines and skip invalid ones
        expected_rules = {
            "ベッドロック": "Bedrock",
            "エス3": "S3",
        }
        self._test_rules_file_loading(rules_content, expected_rules)

    def test_replacement_rules_file_not_found(self) -> None:
        """Test handling of non-existent replacement rules file."""
        config = PreprocessingConfig()

        with pytest.raises(FileNotFoundError):
            config._load_replacement_rules_from_file("/nonexistent/file.txt")

    def test_integration_with_preprocessing(self) -> None:
        """Test word replacement integration with full preprocessing pipeline."""
        config = PreprocessingConfig(
            replacement_rules={"ベッドロック": "Bedrock", "エス3": "S3"},
            enable_word_replacement=True,
            min_duration=0.1,  # Low threshold for testing
        )
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                start_time="00:00:00.000",
                end_time="00:00:02.000",
                text="ベッドロックとエス3について説明します。",
                speaker="Speaker1",
            ),
        ]

        processed_cues = preprocessor.preprocess_cues(cues)
        assert len(processed_cues) == 1
        assert "Bedrock" in processed_cues[0].text
        assert "S3" in processed_cues[0].text
        assert "ベッドロック" not in processed_cues[0].text
        assert "エス3" not in processed_cues[0].text

"""Tests for word replacement functionality."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from vtt2minutes.parser import VTTCue
from vtt2minutes.preprocessor import PreprocessingConfig, TextPreprocessor


class TestWordReplacement:
    """Test word replacement functionality."""

    def test_basic_replacement(self) -> None:
        """Test basic word replacement functionality."""
        config = PreprocessingConfig(
            replacement_rules={"ベッドロック": "Bedrock", "エス3": "S3"},
            enable_word_replacement=True,
        )
        preprocessor = TextPreprocessor(config)

        # Test replacement in VTT cue
        cue = VTTCue(
            start_time="00:00:00.000",
            end_time="00:00:02.000",
            text="今日はベッドロックとエス3について話します。",
            speaker="Speaker1",
        )

        result = preprocessor._apply_word_replacement(cue.text)
        assert result == "今日はBedrockとS3について話します。"

    def test_replacement_order(self) -> None:
        """Test that longer words are replaced before shorter ones."""
        config = PreprocessingConfig(
            replacement_rules={
                "アマゾンウェブサービス": "Amazon Web Services",
                "アマゾン": "Amazon",
            },
            enable_word_replacement=True,
        )
        preprocessor = TextPreprocessor(config)

        # Test that the longer phrase is replaced, not just "アマゾン"
        result = preprocessor._apply_word_replacement(
            "アマゾンウェブサービスを使用します。"
        )
        assert result == "Amazon Web Servicesを使用します。"

    def test_no_replacement_when_disabled(self) -> None:
        """Test that replacement is skipped when disabled."""
        config = PreprocessingConfig(
            replacement_rules={"ベッドロック": "Bedrock"},
            enable_word_replacement=False,
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor._apply_word_replacement("ベッドロックについて")
        assert result == "ベッドロックについて"

    def test_empty_replacement_rules(self) -> None:
        """Test behavior with empty replacement rules."""
        config = PreprocessingConfig(
            replacement_rules={},
            enable_word_replacement=True,
        )
        preprocessor = TextPreprocessor(config)

        original_text = "ベッドロックについて話します。"
        result = preprocessor._apply_word_replacement(original_text)
        assert result == original_text

    def test_replacement_rules_file_loading(self) -> None:
        """Test loading replacement rules from file."""
        rules_content = """# Test replacement rules
ベッドロック -> Bedrock
エス3 -> S3
# Comment line
アマゾン -> Amazon

# Empty line above is ignored
"""

        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(rules_content)
            rules_file = Path(f.name)

        try:
            config = PreprocessingConfig(replacement_rules_file=rules_file)
            expected_rules = {
                "ベッドロック": "Bedrock",
                "エス3": "S3",
                "アマゾン": "Amazon",
            }
            assert config.replacement_rules == expected_rules
        finally:
            rules_file.unlink()

    def test_replacement_rules_file_invalid_format(self) -> None:
        """Test handling of invalid format in replacement rules file."""
        rules_content = """# Test replacement rules
ベッドロック -> Bedrock
invalid_line_without_arrow
エス3 -> S3
"""

        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(rules_content)
            rules_file = Path(f.name)

        try:
            # Should load valid lines and skip invalid ones
            config = PreprocessingConfig(replacement_rules_file=rules_file)
            expected_rules = {
                "ベッドロック": "Bedrock",
                "エス3": "S3",
            }
            assert config.replacement_rules == expected_rules
        finally:
            rules_file.unlink()

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

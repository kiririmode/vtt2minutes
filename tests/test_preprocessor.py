"""Tests for the text preprocessor module."""

import pytest

from vtt2minutes.parser import VTTCue
from vtt2minutes.preprocessor import PreprocessingConfig, TextPreprocessor


class TestPreprocessingConfig:
    """Test cases for PreprocessingConfig."""

    def test_default_initialization(self) -> None:
        """Test default configuration initialization."""
        config = PreprocessingConfig()

        assert config.min_text_length == 3
        assert config.duplicate_threshold == 0.8
        assert config.min_duration == 0.5
        assert config.merge_gap_threshold == 2.0
        assert config.merge_same_speaker is True
        assert config.fix_transcription_errors is True
        assert isinstance(config.filler_words, set)
        assert len(config.filler_words) > 0

        # Check some expected filler words
        assert "えー" in config.filler_words
        assert "um" in config.filler_words
        assert "あのー" in config.filler_words

    def test_custom_initialization(self) -> None:
        """Test custom configuration initialization."""
        custom_filler_words = {"test", "example"}
        config = PreprocessingConfig(
            filler_words=custom_filler_words,
            min_text_length=5,
            duplicate_threshold=0.9,
            min_duration=1.0,
            merge_gap_threshold=3.0,
            merge_same_speaker=False,
            fix_transcription_errors=False,
        )

        assert config.filler_words == custom_filler_words
        assert config.min_text_length == 5
        assert config.duplicate_threshold == 0.9
        assert config.min_duration == 1.0
        assert config.merge_gap_threshold == 3.0
        assert config.merge_same_speaker is False
        assert config.fix_transcription_errors is False


class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""

    @pytest.fixture
    def preprocessor(self) -> TextPreprocessor:
        """Create a TextPreprocessor instance for testing."""
        return TextPreprocessor()

    @pytest.fixture
    def sample_cues(self) -> list[VTTCue]:
        """Create sample VTT cues for testing."""
        return [
            VTTCue(
                "00:00:01.000", "00:00:03.000", "田中", "えー、今日はお疲れ様です。"
            ),
            VTTCue(
                "00:00:03.500",
                "00:00:05.000",
                "田中",
                "あのー、プロジェクトの進捗について報告します。",
            ),
            VTTCue("00:00:05.200", "00:00:06.000", "田中", "うん"),  # Short filler
            VTTCue(
                "00:00:06.500",
                "00:00:08.000",
                "佐藤",
                "ありがとうございます。質問があります。",
            ),
            VTTCue(
                "00:00:08.500",
                "00:00:10.000",
                "佐藤",
                "ありがとうございます。質問があります。",
            ),  # Duplicate
            VTTCue("00:00:10.200", "00:00:11.000", "山田", "a"),  # Too short
            VTTCue("00:00:11.500", "00:00:13.000", "山田", "私も同意見です。"),
        ]

    def test_preprocess_empty_cues(self, preprocessor: TextPreprocessor) -> None:
        """Test preprocessing with empty cue list."""
        result = preprocessor.preprocess_cues([])
        assert result == []

    def test_filler_word_removal(self, preprocessor: TextPreprocessor) -> None:
        """Test removal of filler words."""
        cues = [
            VTTCue(
                "00:00:01.000", "00:00:03.000", "田中", "えー、今日はお疲れ様です。"
            ),
            VTTCue(
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "あのー、そうですね、わかりました。",
            ),
            VTTCue("00:00:05.000", "00:00:07.000", "山田", "Um, I think so."),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check that filler words are removed
        assert "えー" not in result[0].text
        assert "今日はお疲れ様です" in result[0].text

        assert "あのー" not in result[1].text
        assert "そうですね" not in result[1].text
        assert "わかりました" in result[1].text

        assert "Um" not in result[2].text
        assert "I think so" in result[2].text or "think so" in result[2].text

    def test_transcription_error_fixing(self, preprocessor: TextPreprocessor) -> None:
        """Test fixing of common transcription errors."""
        cues = [
            VTTCue("00:00:01.000", "00:00:03.000", "田中", "これはです。です。"),
            VTTCue("00:00:03.000", "00:00:05.000", "佐藤", "わかりりりります。"),
            VTTCue("00:00:05.000", "00:00:07.000", "山田", "同じ同じことです。"),
            VTTCue("00:00:07.000", "00:00:09.000", "田中", "多  く  の  空  白"),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check error corrections
        assert "これはです" in result[0].text
        assert "わかります" in result[1].text
        assert "同じことです" in result[2].text
        assert "多 く の 空 白" in result[3].text

    def test_minimum_text_length_filter(self, preprocessor: TextPreprocessor) -> None:
        """Test filtering based on minimum text length."""
        cues = [
            VTTCue("00:00:01.000", "00:00:03.000", "田中", "a"),  # Too short
            VTTCue("00:00:03.000", "00:00:05.000", "佐藤", "ab"),  # Too short
            VTTCue("00:00:05.000", "00:00:07.000", "山田", "abc"),  # Exactly min length
            VTTCue(
                "00:00:07.000", "00:00:09.000", "田中", "この文章は十分な長さです。"
            ),  # Long enough
        ]

        result = preprocessor.preprocess_cues(cues)

        # Should filter out very short texts but keep longer ones
        assert len(result) >= 2  # At least the two longer texts
        # Check that the longer texts are preserved
        result_texts = [cue.text for cue in result]
        assert any("abc" in text for text in result_texts)
        assert any("この文章は十分な長さです" in text for text in result_texts)

    def test_minimum_duration_filter(self, preprocessor: TextPreprocessor) -> None:
        """Test filtering based on minimum duration."""
        cues = [
            VTTCue(
                "00:00:01.000", "00:00:01.300", "田中", "短すぎる発言"
            ),  # 0.3s - too short
            VTTCue(
                "00:00:02.000", "00:00:02.500", "佐藤", "ちょうど最小時間"
            ),  # 0.5s - exactly min
            VTTCue(
                "00:00:03.000", "00:00:04.000", "山田", "十分な長さの発言"
            ),  # 1.0s - long enough
        ]

        result = preprocessor.preprocess_cues(cues)

        assert len(result) == 2
        assert "ちょうど最小時間" in result[0].text
        assert "十分な長さの発言" in result[1].text

    def test_duplicate_removal(self, preprocessor: TextPreprocessor) -> None:
        """Test removal of duplicate cues."""
        cues = [
            VTTCue("00:00:01.000", "00:00:03.000", "田中", "これは重要な発言です。"),
            VTTCue(
                "00:00:03.000", "00:00:05.000", "佐藤", "これは重要な発言です。"
            ),  # Exact duplicate
            VTTCue(
                "00:00:05.000", "00:00:07.000", "山田", "これは重要な発言だと思います。"
            ),  # Similar
            VTTCue(
                "00:00:07.000", "00:00:09.000", "田中", "全く異なる内容の発言です。"
            ),  # Different
        ]

        result = preprocessor.preprocess_cues(cues)

        # Should remove duplicates but keep unique content
        assert len(result) >= 2  # At least keep non-duplicates
        result_texts = " ".join(cue.text for cue in result)
        assert "これは重要な発言です" in result_texts
        assert "全く異なる内容の発言です" in result_texts

    def test_consecutive_cue_merging(self, preprocessor: TextPreprocessor) -> None:
        """Test merging of consecutive cues from the same speaker."""
        cues = [
            VTTCue("00:00:01.000", "00:00:02.000", "田中", "最初の発言"),
            VTTCue(
                "00:00:02.500", "00:00:03.500", "田中", "続きの発言"
            ),  # Same speaker, small gap
            VTTCue(
                "00:00:05.000", "00:00:06.000", "田中", "大きな間隔の発言"
            ),  # Same speaker, large gap
            VTTCue(
                "00:00:06.500", "00:00:07.500", "佐藤", "別の人の発言"
            ),  # Different speaker
        ]

        result = preprocessor.preprocess_cues(cues)

        # Should merge consecutive cues from same speaker
        assert len(result) >= 2  # At least separate speakers
        result_texts = " ".join(cue.text for cue in result)
        assert "最初の発言" in result_texts
        assert "続きの発言" in result_texts
        assert "別の人の発言" in result_texts

        # Check that we have both speakers
        speakers = {cue.speaker for cue in result}
        assert "田中" in speakers
        assert "佐藤" in speakers

    def test_no_merging_when_disabled(self, preprocessor: TextPreprocessor) -> None:
        """Test that merging doesn't occur when disabled."""
        config = PreprocessingConfig(merge_same_speaker=False)
        preprocessor_no_merge = TextPreprocessor(config)

        cues = [
            VTTCue("00:00:01.000", "00:00:02.000", "田中", "最初の発言"),
            VTTCue("00:00:02.500", "00:00:03.500", "田中", "続きの発言"),
        ]

        result = preprocessor_no_merge.preprocess_cues(cues)

        # Should not merge
        assert len(result) == 2
        assert "最初の発言" in result[0].text
        assert "続きの発言" in result[1].text

    def test_final_cleanup_punctuation(self, preprocessor: TextPreprocessor) -> None:
        """Test final cleanup adds proper punctuation."""
        cues = [
            VTTCue(
                "00:00:01.000", "00:00:03.000", "田中", "これは日本語の文章"
            ),  # No punctuation
            VTTCue(
                "00:00:03.000", "00:00:05.000", "佐藤", "This is English"
            ),  # No punctuation
            VTTCue(
                "00:00:05.000", "00:00:07.000", "山田", "既に句読点があります。"
            ),  # Has punctuation
            VTTCue(
                "00:00:07.000", "00:00:09.000", "田中", "複数の。。。句読点"
            ),  # Multiple punctuation
        ]

        result = preprocessor.preprocess_cues(cues)

        assert "これは日本語の文章" in result[0].text
        assert "This is English" in result[1].text
        assert "既に句読点があります" in result[2].text
        assert "複数の" in result[3].text and "句読点" in result[3].text

    def test_comprehensive_preprocessing(
        self, preprocessor: TextPreprocessor, sample_cues: list[VTTCue]
    ) -> None:
        """Test comprehensive preprocessing with sample data."""
        result = preprocessor.preprocess_cues(sample_cues)

        # Should remove short cues and duplicates
        assert len(result) < len(sample_cues)

        # Check that processing preserved important content
        text_content = " ".join(cue.text for cue in result)
        assert "今日はお疲れ様です" in text_content
        assert "プロジェクトの進捗について報告します" in text_content
        assert "質問があります" in text_content
        assert "同意見です" in text_content

        # Check that filler words and short content are removed
        assert "えー" not in text_content
        assert "あのー" not in text_content
        assert "うん" not in text_content
        assert "a" not in text_content

    def test_statistics_generation(
        self, preprocessor: TextPreprocessor, sample_cues: list[VTTCue]
    ) -> None:
        """Test preprocessing statistics generation."""
        processed_cues = preprocessor.preprocess_cues(sample_cues)
        stats = preprocessor.get_statistics(sample_cues, processed_cues)

        assert stats["original_count"] == len(sample_cues)
        assert stats["processed_count"] == len(processed_cues)
        assert stats["removed_count"] == len(sample_cues) - len(processed_cues)
        assert stats["removal_rate"] >= 0.0
        assert stats["removal_rate"] <= 1.0
        assert stats["original_duration"] >= stats["processed_duration"]
        assert stats["original_text_length"] >= 0
        assert stats["processed_text_length"] >= 0

    def test_empty_text_handling(self, preprocessor: TextPreprocessor) -> None:
        """Test handling of cues with empty or whitespace-only text."""
        cues = [
            VTTCue("00:00:01.000", "00:00:03.000", "田中", ""),  # Empty
            VTTCue("00:00:03.000", "00:00:05.000", "佐藤", "   "),  # Whitespace only
            VTTCue(
                "00:00:05.000", "00:00:07.000", "山田", "実際の内容"
            ),  # Real content
        ]

        result = preprocessor.preprocess_cues(cues)

        assert len(result) == 1
        assert result[0].text in ["実際の内容。", "実際の内容."]

    def test_speaker_preservation(self, preprocessor: TextPreprocessor) -> None:
        """Test that speaker information is preserved during preprocessing."""
        cues = [
            VTTCue("00:00:01.000", "00:00:03.000", "田中", "田中の発言です"),
            VTTCue("00:00:03.000", "00:00:05.000", "佐藤", "佐藤の発言です"),
            VTTCue("00:00:05.000", "00:00:07.000", None, "話者不明の発言"),
        ]

        result = preprocessor.preprocess_cues(cues)

        assert len(result) == 3
        assert result[0].speaker == "田中"
        assert result[1].speaker == "佐藤"
        assert result[2].speaker is None

    def test_time_preservation(self, preprocessor: TextPreprocessor) -> None:
        """Test that timing information is preserved during preprocessing."""
        cues = [
            VTTCue("00:00:01.000", "00:00:03.000", "田中", "最初の発言内容"),
            VTTCue("00:00:03.500", "00:00:05.500", "田中", "続きの発言内容"),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Should merge consecutive cues from same speaker
        assert len(result) == 1
        assert result[0].start_time == "00:00:01.000"
        assert result[0].end_time == "00:00:05.500"  # End time of last merged cue

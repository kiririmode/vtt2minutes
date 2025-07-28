"""Tests for the text preprocessor module."""

import tempfile
from collections.abc import Sequence
from pathlib import Path

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

    def _create_test_cues(
        self, cue_data: Sequence[tuple[str, str, str | None, str]]
    ) -> list[VTTCue]:
        """Helper to create VTTCue objects from tuple data."""
        return [
            VTTCue(start, end, speaker, text) for start, end, speaker, text in cue_data
        ]

    def _test_preprocessing(
        self,
        preprocessor: TextPreprocessor,
        input_cues: list[VTTCue],
        expected_count: int | None = None,
        expected_texts: list[str] | None = None,
        expected_speakers: list[str | None] | None = None,
    ) -> list[VTTCue]:
        """Helper to run preprocessing and verify results."""
        result = preprocessor.preprocess_cues(input_cues)

        if expected_count is not None:
            assert len(result) == expected_count

        if expected_texts is not None:
            actual_texts = [cue.text for cue in result]
            for expected_text in expected_texts:
                assert expected_text in actual_texts

        if expected_speakers is not None:
            actual_speakers = [cue.speaker for cue in result]
            for expected_speaker in expected_speakers:
                assert expected_speaker in actual_speakers

        return result

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
        cue_data = [
            ("00:00:01.000", "00:00:03.000", "田中", "えー、今日はお疲れ様です。"),
            (
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "あのー、そうですね、わかりました。",
            ),
        ]
        cues = self._create_test_cues(cue_data)
        result = self._test_preprocessing(preprocessor, cues, expected_count=2)

        # Check that filler words are removed
        assert "えー" not in result[0].text
        assert "今日はお疲れ様です" in result[0].text

        assert "あのー" not in result[1].text
        assert "そうですね" not in result[1].text
        assert "わかりました" in result[1].text

    def test_transcription_error_fixing(self, preprocessor: TextPreprocessor) -> None:
        """Test fixing of common transcription errors."""
        cue_data = [
            ("00:00:01.000", "00:00:03.000", "田中", "これはです。です。"),
            ("00:00:03.000", "00:00:05.000", "佐藤", "わかりりりります。"),
            ("00:00:05.000", "00:00:07.000", "山田", "同じ同じことです。"),
            ("00:00:07.000", "00:00:09.000", "田中", "多  く  の  空  白"),
        ]
        cues = self._create_test_cues(cue_data)
        result = self._test_preprocessing(preprocessor, cues, expected_count=4)

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

    def test_no_merging_when_disabled(self) -> None:
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
                "00:00:05.000", "00:00:07.000", "山田", "既に句読点があります。"
            ),  # Has punctuation
            VTTCue(
                "00:00:07.000", "00:00:09.000", "田中", "複数の。。。句読点"
            ),  # Multiple punctuation
        ]

        result = preprocessor.preprocess_cues(cues)

        assert "これは日本語の文章" in result[0].text
        assert "既に句読点があります" in result[1].text
        assert "複数の" in result[2].text and "句読点" in result[2].text

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
        cue_data = [
            ("00:00:01.000", "00:00:03.000", "田中", "田中の発言です"),
            ("00:00:03.000", "00:00:05.000", "佐藤", "佐藤の発言です"),
            ("00:00:05.000", "00:00:07.000", None, "話者不明の発言"),
        ]
        cues = self._create_test_cues(cue_data)
        result = self._test_preprocessing(
            preprocessor,
            cues,
            expected_count=3,
            expected_speakers=["田中", "佐藤", None],
        )

        assert result[0].speaker == "田中"
        assert result[1].speaker == "佐藤"
        assert result[2].speaker is None

    def test_time_preservation(self, preprocessor: TextPreprocessor) -> None:
        """Test that timing information is preserved during preprocessing."""
        cue_data = [
            ("00:00:01.000", "00:00:03.000", "田中", "最初の発言内容"),
            ("00:00:03.500", "00:00:05.500", "田中", "続きの発言内容"),
        ]
        cues = self._create_test_cues(cue_data)
        result = self._test_preprocessing(preprocessor, cues, expected_count=1)

        # Should merge consecutive cues from same speaker
        assert result[0].start_time == "00:00:01.000"
        assert result[0].end_time == "00:00:05.500"  # End time of last merged cue


class TestFilterWordsFile:
    """Test cases for filter words file functionality."""

    def _create_temp_filter_file(self, content: list[str]) -> Path:
        """Helper to create temporary filter file with given content."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for line in content:
                f.write(line + "\n")
            return Path(f.name)

    def _test_filter_file_loading(
        self, file_content: list[str], expected_words: set[str]
    ) -> None:
        """Helper to test filter file loading with given content and expected result."""
        temp_path = self._create_temp_filter_file(file_content)

        try:
            config = PreprocessingConfig(filler_words_file=temp_path)
            assert config.filler_words == expected_words
        finally:
            temp_path.unlink()

    def test_load_filler_words_from_file(self) -> None:
        """Test loading filler words from a file."""
        file_content = [
            "# Test filter words file",
            "えー",
            "um",
            "",  # Empty line
            "# Another comment",
            "test_word",
        ]
        expected_words = {"えー", "um", "test_word"}
        self._test_filter_file_loading(file_content, expected_words)

    def test_load_filler_words_file_not_found(self) -> None:
        """Test error handling when filter words file doesn't exist."""
        non_existent_path = Path("non_existent_filter_words.txt")

        with pytest.raises(FileNotFoundError, match="Filler words file not found"):
            PreprocessingConfig(filler_words_file=non_existent_path)

    def test_load_filler_words_empty_file(self) -> None:
        """Test loading from an empty filter words file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Only comments\n")
            f.write("\n")
            f.write("# No actual words\n")
            temp_path = Path(f.name)

        try:
            config = PreprocessingConfig(filler_words_file=temp_path)
            assert config.filler_words == set()
        finally:
            temp_path.unlink()

    def test_load_filler_words_with_unicode(self) -> None:
        """Test loading Japanese and other unicode filler words."""
        file_content = [
            "あのー",
            "そうですね",
            "café",  # With accent
            "naïve",  # With diaeresis
        ]
        expected_words = {"あのー", "そうですね", "café", "naïve"}
        self._test_filter_file_loading(file_content, expected_words)

    def test_filler_words_file_overrides_default(self) -> None:
        """Test that filler words file overrides default words."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("custom_word1\n")
            f.write("custom_word2\n")
            temp_path = Path(f.name)

        try:
            config = PreprocessingConfig(filler_words_file=temp_path)

            # Should only contain custom words, not defaults
            assert config.filler_words == {"custom_word1", "custom_word2"}
            assert config.filler_words is not None
            assert "えー" not in config.filler_words  # Default Japanese word
            assert "um" not in config.filler_words  # Default English word
        finally:
            temp_path.unlink()

    def test_preprocessor_with_custom_filter_file(self) -> None:
        """Test text preprocessor using custom filter words file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("custom_filler\n")
            f.write("test_word\n")
            temp_path = Path(f.name)

        try:
            config = PreprocessingConfig(filler_words_file=temp_path)
            preprocessor = TextPreprocessor(config)

            cues = [
                VTTCue(
                    "00:00:01.000",
                    "00:00:03.000",
                    "田中",
                    "custom_filler、これはテストです。",
                ),
                VTTCue(
                    "00:00:03.000",
                    "00:00:05.000",
                    "佐藤",
                    "test_word、重要な内容です。",
                ),
            ]

            result = preprocessor.preprocess_cues(cues)

            # Custom filler words should be removed
            assert "custom_filler" not in result[0].text
            assert "test_word" not in result[1].text
            # Content should remain
            assert "これはテストです" in result[0].text
            assert "重要な内容です" in result[1].text
        finally:
            temp_path.unlink()


class TestJapaneseTextProcessing:
    """Test cases for Japanese text processing and spacing improvements."""

    def test_japanese_filler_word_removal_and_spacing(self) -> None:
        """Test that Japanese filler word removal and spacing works correctly."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                "00:00:01.000",
                "00:00:03.000",
                "田中",
                "えー、おはようございます、あのー、今日の会議を、えっと、始めたいと思います。",
            ),
            VTTCue(
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "おはようございます、あ、よろしくお願いします。はい。",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check that filler words are removed
        assert "えー" not in result[0].text
        assert "あのー" not in result[0].text
        assert "えっと" not in result[0].text
        assert "おはようございます" in result[0].text
        assert "今日の会議を" in result[0].text
        assert "始めたいと思います" in result[0].text

        # Check that sentence endings are preserved
        assert "。" in result[0].text  # Periods should be preserved

        # Check second cue
        assert "はい" not in result[1].text
        assert "おはようございます" in result[1].text
        assert "よろしくお願いします" in result[1].text

        # Most importantly: check that spacing is natural (not excessive)
        for cue in result:
            # Should not have multiple consecutive spaces
            assert "  " not in cue.text
            # Should not have unnatural spacing patterns
            assert not cue.text.startswith(" ")
            assert not cue.text.endswith(" ")

    def test_japanese_text_no_unnatural_spaces(self) -> None:
        """Test that Japanese text doesn't have unnatural spacing after processing."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                "00:00:01.000",
                "00:00:03.000",
                "田中",
                "あー、すみません、今、音声が、ちょっと、えっと、遅れてまして…。",
            ),
            VTTCue(
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "えー、Aプロジェクトは、予定、予定通りに、進んでます。",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check that unnatural spaces are not inserted
        for cue in result:
            # Should not have multiple consecutive spaces
            assert "  " not in cue.text
            # Should not start with punctuation (leading comma issue)
            assert not cue.text.startswith("、")
            assert not cue.text.startswith("，")
            # Should end with proper punctuation
            assert cue.text.endswith(("。", "…", ".", "!", "?", "！", "？"))
            # Should not have unnatural whitespace patterns
            assert not cue.text.startswith(" ")
            assert not cue.text.endswith(" ")
            # Check that some meaningful content is preserved
            assert len(cue.text) > 5  # Should have substantial content

    def test_mixed_japanese_processing(self) -> None:
        """Test processing of Japanese content with technical terms."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                "00:00:01.000",
                "00:00:03.000",
                "田中",
                "えー、Bプロジェクト、Bプロジェクトなんですけど、仕様が、途中で変わったので…。",
            ),
            VTTCue(
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "APIの仕様について、えー、いいと思います。",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check Japanese processing
        assert "えー" not in result[0].text
        assert "Bプロジェクト" in result[0].text
        assert "仕様が" in result[0].text
        assert "、" in result[0].text

        # Check mixed content processing
        assert "えー" not in result[1].text
        assert "API" in result[1].text
        assert "いいと思います" in result[1].text

    def test_complex_japanese_sentence_structure(self) -> None:
        """Test complex Japanese sentences with multiple clauses."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                "00:00:01.000",
                "00:00:03.000",
                "田中",
                "あの、仕様変更が、ありまして、で、その、影響で、ちょっと遅れが…。",
            ),
            VTTCue(
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "なるほど、なるほど、それは、来週までに、あの、対策できますか？",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check that complex sentence structure is preserved
        for cue in result:
            # Should maintain natural flow
            assert "仕様変更が" in result[0].text or "ありまして" in result[0].text
            assert "なるほど" in result[1].text
            assert "来週までに" in result[1].text
            assert "対策できますか" in result[1].text

            # Should not have unnatural artifacts
            assert not cue.text.startswith("、")
            assert "、、" not in cue.text
            assert "。。" not in cue.text

    def test_japanese_vs_english_space_handling(self) -> None:
        """Test that Japanese and English text are handled differently for spacing."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                "00:00:01.000",
                "00:00:03.000",
                "田中",
                "ただ、ちょっとだけ、進捗が、あの、遅れそうでして、うーん、追加の、作業が必要かと…。",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Japanese text should have natural punctuation without excess spaces
        japanese_text = result[0].text
        assert "、" in japanese_text
        assert "進捗が" in japanese_text
        assert "追加の" in japanese_text
        assert "作業が必要かと" in japanese_text
        # Should not have unnatural consecutive spaces in Japanese
        assert "  " not in japanese_text

    def test_punctuation_boundary_cases(self) -> None:
        """Test edge cases with punctuation and filler word boundaries."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        cues = [
            VTTCue(
                "00:00:01.000",
                "00:00:03.000",
                "田中",
                "えー、じゃあ、最初に、先週の、えー、タスクの、えー、進捗を…報告お願いします。",
            ),
            VTTCue(
                "00:00:03.000",
                "00:00:05.000",
                "佐藤",
                "ありがとうございます。では、じゃあ、その、アクション、アイテムを、整理しましょう。",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Check that leading punctuation is handled correctly
        for cue in result:
            # Should not start with comma or space
            assert not cue.text.startswith("、")
            assert not cue.text.startswith(" ")
            # Should not have doubled punctuation
            assert "、、" not in cue.text
            assert "。。" not in cue.text
            # Should not have multiple consecutive spaces
            assert "  " not in cue.text

        # Check that meaningful content is preserved in each cue
        text1 = result[0].text
        assert any(
            word in text1 for word in ["最初に", "先週", "タスク", "進捗", "報告"]
        )

        text2 = result[1].text
        assert any(
            word in text2
            for word in [
                "ありがとうございます",
                "アクション",
                "アイテム",
                "整理しましょう",
            ]
        )

    def test_original_vtt_sample_processing(self) -> None:
        """Test processing that matches the original sample.vtt file content."""
        config = PreprocessingConfig()
        preprocessor = TextPreprocessor(config)

        # Actual content from the sample.vtt file
        cues = [
            VTTCue(
                "00:00:00.000",
                "00:00:05.000",
                "田中 一郎",
                "田中 一郎: えー、おはようございます、あのー、今日の会議を、"
                "えっと、始めたいと思います。",
            ),
            VTTCue(
                "00:00:05.500",
                "00:00:11.000",
                "佐藤 花子",
                "佐藤 花子: おはようございます、あ、よろしくお願いします。",
            ),
            VTTCue(
                "00:00:10.800",
                "00:00:14.000",
                "鈴木 太郎",
                "鈴木 太郎: あ、すみません、今、音声が、ちょっと、"
                "えっと、遅れてまして…。",
            ),
        ]

        result = preprocessor.preprocess_cues(cues)

        # Verify that the result looks natural and properly processed
        assert len(result) == 3

        # Check that speakers are preserved from the text extraction
        assert result[0].speaker == "田中 一郎"
        assert result[1].speaker == "佐藤 花子"
        assert result[2].speaker == "鈴木 太郎"

        # Check that filler words are removed but structure is preserved
        text1 = result[0].text
        assert "えー" not in text1
        assert "あのー" not in text1
        assert "えっと" not in text1
        assert "おはようございます" in text1
        assert "今日の会議" in text1
        assert "始めたいと思います" in text1

        text2 = result[1].text
        assert "おはようございます" in text2
        assert "よろしくお願いします" in text2

        text3 = result[2].text
        assert "ちょっと" not in text3
        assert "えっと" not in text3
        assert "すみません" in text3
        assert "音声が" in text3
        assert "遅れてまして" in text3

        # Verify natural punctuation is maintained
        for cue in result:
            # Should have appropriate punctuation
            assert any(p in cue.text for p in ["、", "。", "…"])
            # Should not have leading punctuation
            assert not cue.text.startswith("、")
            # Should not have doubled spaces
            assert "  " not in cue.text

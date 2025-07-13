"""Tests for the VTT parser module."""

from pathlib import Path

import pytest

from vtt2minutes.parser import VTTCue, VTTParser


class TestVTTCue:
    """Test cases for VTTCue class."""

    def test_cue_creation(self) -> None:
        """Test VTTCue creation and basic properties."""
        cue = VTTCue("00:01:30.500", "00:01:33.200", "田中", "これはテストです。")

        assert cue.start_time == "00:01:30.500"
        assert cue.end_time == "00:01:33.200"
        assert cue.speaker == "田中"
        assert cue.text == "これはテストです。"

    def test_time_to_seconds_conversion(self) -> None:
        """Test conversion of time strings to seconds."""
        cue = VTTCue("00:01:30.500", "00:01:33.200", "田中", "テスト")

        assert cue.start_seconds == 90.5  # 1 min 30.5 sec
        assert cue.end_seconds == 93.2  # 1 min 33.2 sec
        assert (
            abs(cue.duration - 2.7) < 0.01
        )  # 2.7 sec difference (with tolerance for float precision)

    def test_time_to_seconds_mm_ss_format(self) -> None:
        """Test conversion with MM:SS.mmm format."""
        cue = VTTCue("01:30.500", "01:33.200", "田中", "テスト")

        assert cue.start_seconds == 90.5
        assert cue.end_seconds == 93.2

    def test_invalid_time_format(self) -> None:
        """Test handling of invalid time format."""
        cue = VTTCue("invalid", "00:01:30.500", "田中", "テスト")

        with pytest.raises(ValueError, match="Invalid time format"):
            _ = cue.start_seconds


class TestVTTParser:
    """Test cases for VTTParser class."""

    def _parse_and_assert_count(
        self, parser: VTTParser, content: str, expected_count: int
    ) -> list:
        """Helper method to parse content and assert cue count."""
        cues = parser.parse_content(content)
        assert len(cues) == expected_count
        return cues

    def _assert_single_cue_properties(
        self, cue, expected_text: str | None = None, expected_speaker: str | None = None
    ) -> None:
        """Helper method to assert properties of a single cue."""
        if expected_text is not None:
            assert cue.text == expected_text
        if expected_speaker is not None:
            assert cue.speaker == expected_speaker

    @pytest.fixture
    def parser(self) -> VTTParser:
        """Create a VTTParser instance for testing."""
        return VTTParser()

    @pytest.fixture
    def sample_vtt_content(self) -> str:
        """Sample VTT content for testing."""
        return """WEBVTT

00:11.000 --> 00:13.000
<v Roger Bingham>We are in New York City

00:13.000 --> 00:16.000
<v Roger Bingham>We're actually at the Lucern Hotel

00:16.000 --> 00:18.000
<v Neil deGrasse Tyson>And with me is Neil deGrasse Tyson

00:18.000 --> 00:20.000
This is a cue without speaker tags

NOTE This is a comment line

00:20.000 --> 00:22.000
<v Neil deGrasse Tyson>Astrophysicist, Director of the Hayden Planetarium
"""

    @pytest.fixture
    def japanese_vtt_content(self) -> str:
        """Japanese VTT content for testing."""
        return """WEBVTT

00:00:01.000 --> 00:00:03.000
<v 田中>おはようございます。今日はよろしくお願いします。

00:00:03.500 --> 00:00:05.000
<v 佐藤>こちらこそ、よろしくお願いします。

00:00:05.500 --> 00:00:08.000
<v 田中>それでは、プロジェクトの進捗について報告させていただきます。

00:00:08.500 --> 00:00:10.000
質問があります。
"""

    def test_parse_empty_content(self, parser: VTTParser) -> None:
        """Test parsing empty content."""
        with pytest.raises(ValueError, match="Invalid VTT file: missing WEBVTT header"):
            parser.parse_content("")

    def test_parse_invalid_header(self, parser: VTTParser) -> None:
        """Test parsing content without WEBVTT header."""
        content = "This is not a VTT file"
        with pytest.raises(ValueError, match="Invalid VTT file: missing WEBVTT header"):
            parser.parse_content(content)

    def test_parse_basic_vtt(self, parser: VTTParser, sample_vtt_content: str) -> None:
        """Test parsing basic VTT content."""
        cues = parser.parse_content(sample_vtt_content)

        assert len(cues) == 5

        # Check first cue
        assert cues[0].start_time == "00:11.000"
        assert cues[0].end_time == "00:13.000"
        assert cues[0].speaker == "Roger Bingham"
        assert cues[0].text == "We are in New York City"

        # Check cue without speaker
        assert cues[3].speaker is None
        assert cues[3].text == "This is a cue without speaker tags"

        # Check that NOTE lines are ignored
        # (should be 5 cues, not 6 if NOTE was processed)
        assert len(cues) == 5

    def test_parse_japanese_vtt(
        self, parser: VTTParser, japanese_vtt_content: str
    ) -> None:
        """Test parsing Japanese VTT content."""
        cues = parser.parse_content(japanese_vtt_content)

        assert len(cues) == 4

        # Check Japanese speaker names
        assert cues[0].speaker == "田中"
        assert cues[1].speaker == "佐藤"
        assert cues[2].speaker == "田中"
        assert cues[3].speaker is None

        # Check Japanese text
        assert "おはようございます" in cues[0].text
        assert "よろしくお願いします" in cues[1].text
        assert "プロジェクトの進捗" in cues[2].text
        assert "質問があります" in cues[3].text

    def test_html_tag_removal(self, parser: VTTParser) -> None:
        """Test removal of HTML tags from text."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Speaker>This has <b>bold</b> and <i>italic</i> text
"""
        cues = self._parse_and_assert_count(parser, content, 1)
        self._assert_single_cue_properties(
            cues[0], "This has bold and italic text", "Speaker"
        )

    def test_multiline_cue_text(self, parser: VTTParser) -> None:
        """Test parsing cues with multiple lines of text."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Speaker>First line of text
Second line of text
Third line of text
"""
        cues = self._parse_and_assert_count(parser, content, 1)
        self._assert_single_cue_properties(
            cues[0], "First line of text Second line of text Third line of text"
        )

    def test_get_speakers(self, parser: VTTParser, sample_vtt_content: str) -> None:
        """Test extracting unique speakers from cues."""
        cues = parser.parse_content(sample_vtt_content)
        speakers = parser.get_speakers(cues)

        assert len(speakers) == 2
        assert "Neil deGrasse Tyson" in speakers
        assert "Roger Bingham" in speakers
        assert speakers == sorted(speakers)  # Should be sorted

    def test_get_duration(self, parser: VTTParser, sample_vtt_content: str) -> None:
        """Test calculating total duration."""
        cues = parser.parse_content(sample_vtt_content)
        duration = parser.get_duration(cues)

        # Duration should be from first start to last end
        expected_duration = 22.0 - 11.0  # 11 seconds
        assert duration == expected_duration

    def test_get_duration_empty_cues(self, parser: VTTParser) -> None:
        """Test duration calculation with empty cues."""
        duration = parser.get_duration([])
        assert duration == 0.0

    def test_filter_by_speaker(
        self, parser: VTTParser, sample_vtt_content: str
    ) -> None:
        """Test filtering cues by speaker."""
        cues = parser.parse_content(sample_vtt_content)
        roger_cues = parser.filter_by_speaker(cues, "Roger Bingham")

        assert len(roger_cues) == 2
        for cue in roger_cues:
            assert cue.speaker == "Roger Bingham"

    def test_filter_by_time_range(
        self, parser: VTTParser, sample_vtt_content: str
    ) -> None:
        """Test filtering cues by time range."""
        cues = parser.parse_content(sample_vtt_content)
        filtered_cues = parser.filter_by_time_range(cues, 15.0, 19.0)

        # Should include cues that start and end within the range
        assert len(filtered_cues) >= 1  # At least one cue in range
        for cue in filtered_cues:
            assert cue.start_seconds >= 15.0
            assert cue.end_seconds <= 19.0

    def test_parse_file_not_found(self, parser: VTTParser) -> None:
        """Test parsing non-existent file."""
        non_existent_file = Path("non_existent.vtt")

        with pytest.raises(FileNotFoundError):
            parser.parse_file(non_existent_file)

    def test_whitespace_normalization(self, parser: VTTParser) -> None:
        """Test that whitespace is properly normalized."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Speaker>Text   with    multiple     spaces
"""
        cues = self._parse_and_assert_count(parser, content, 1)
        self._assert_single_cue_properties(cues[0], "Text with multiple spaces")

    def test_empty_cue_handling(self, parser: VTTParser) -> None:
        """Test handling of empty cues."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000


00:00:04.000 --> 00:00:06.000
<v Speaker>Actual content
"""
        # Should only have one cue (empty cue should be ignored)
        cues = self._parse_and_assert_count(parser, content, 1)
        self._assert_single_cue_properties(cues[0], "Actual content")

    def test_timing_line_variations(self, parser: VTTParser) -> None:
        """Test various timing line formats."""
        content = """WEBVTT

00:11.000 --> 00:13.000
Text with MM:SS format

00:01:11.000 --> 00:01:13.000
Text with HH:MM:SS format

1:11.000 --> 1:13.000
Text with M:SS format
"""
        cues = parser.parse_content(content)

        assert len(cues) == 3
        assert cues[0].start_seconds == 11.0
        assert cues[1].start_seconds == 71.0  # 1 min 11 sec
        assert cues[2].start_seconds == 71.0  # 1 min 11 sec

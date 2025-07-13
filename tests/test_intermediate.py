"""Tests for intermediate file functionality."""

import tempfile
from pathlib import Path

from vtt2minutes.intermediate import IntermediateTranscriptWriter
from vtt2minutes.parser import VTTCue


class TestIntermediateTranscriptWriter:
    """Test cases for IntermediateTranscriptWriter."""

    def _create_temp_file_and_test_markdown(
        self,
        cues: list[VTTCue],
        title: str,
        metadata: dict[str, str | list[str]] | None = None,
        assertions: list[str] | None = None,
    ) -> None:
        """Helper to create temp file, write markdown, and run assertions."""
        writer = IntermediateTranscriptWriter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            if metadata is not None:
                writer.write_markdown(cues, temp_path, title, metadata)
            else:
                writer.write_markdown(cues, temp_path, title)

            assert temp_path.exists()
            content = temp_path.read_text(encoding="utf-8")

            # Common assertions
            assert f"# {title}" in content
            assert "## 発言記録" in content

            # Custom assertions
            if assertions:
                for assertion in assertions:
                    assert assertion in content

        finally:
            temp_path.unlink()

    def create_sample_cues(self) -> list[VTTCue]:
        """Create sample VTT cues for testing."""
        return [
            VTTCue(
                start_time="00:00:00.000",
                end_time="00:00:03.000",
                text="Hello everyone, welcome to the meeting.",
                speaker="Alice",
            ),
            VTTCue(
                start_time="00:00:03.500",
                end_time="00:00:07.000",
                text="Thank you Alice. Let's start with the agenda.",
                speaker="Bob",
            ),
            VTTCue(
                start_time="00:00:07.500",
                end_time="00:00:12.000",
                text="First item is the project status review.",
                speaker="Alice",
            ),
        ]

    def test_init(self) -> None:
        """Test writer initialization."""
        writer = IntermediateTranscriptWriter()
        assert writer is not None

    def test_write_markdown_basic(self) -> None:
        """Test basic markdown file generation."""
        cues = self.create_sample_cues()
        assertions = [
            "### Alice",
            "### Bob",
            "Hello everyone, welcome to the meeting.",
            "Thank you Alice. Let's start with the agenda.",
        ]
        self._create_temp_file_and_test_markdown(
            cues, "Test Meeting", assertions=assertions
        )

    def test_write_markdown_with_metadata(self) -> None:
        """Test markdown generation with metadata."""
        cues = self.create_sample_cues()
        metadata = {
            "date": "2024-01-15",
            "participants": ["Alice", "Bob", "Charlie"],
            "duration": "00:15:30",
        }
        assertions = [
            "**日時:** 2024-01-15",
            "**参加者:** Alice, Bob, Charlie",
            "**総時間:** 00:15:30",
        ]
        self._create_temp_file_and_test_markdown(
            cues, "Project Meeting", metadata, assertions
        )

    def test_write_markdown_empty_cues(self) -> None:
        """Test markdown generation with empty cue list."""
        # Basic structure assertions are handled by helper method
        self._create_temp_file_and_test_markdown([], "Empty Meeting")

    def test_write_markdown_no_speaker(self) -> None:
        """Test markdown generation with cues that have no speaker."""
        cues = [
            VTTCue(
                start_time="00:00:00.000",
                end_time="00:00:03.000",
                text="System notification message.",
                speaker=None,
            ),
        ]
        assertions = ["### 話者不明", "System notification message."]
        self._create_temp_file_and_test_markdown(
            cues, "System Meeting", assertions=assertions
        )

    def test_generate_markdown_content(self) -> None:
        """Test markdown content generation."""
        writer = IntermediateTranscriptWriter()
        cues = self.create_sample_cues()

        metadata = {"participants": ["Alice", "Bob"], "duration": "00:12:00"}

        content = writer._generate_markdown_content(cues, "Test Meeting", metadata)

        # Check structure
        lines = content.split("\n")
        assert lines[0] == "# Test Meeting"
        assert "**参加者:** Alice, Bob" in content
        assert "**総時間:** 00:12:00" in content
        assert "## 発言記録" in content

    def test_add_speaker_section(self) -> None:
        """Test adding a speaker section to markdown."""
        writer = IntermediateTranscriptWriter()
        lines: list[str] = []

        writer._add_speaker_section(
            lines, "Alice", "00:00:00.000", "00:00:05.000", ["Hello", "How are you?"]
        )

        assert len(lines) == 3
        assert lines[0] == "### Alice (00:00:00.000 - 00:00:05.000)"
        assert lines[1] == "Hello How are you?"
        assert lines[2] == ""

    def test_add_speaker_section_no_speaker(self) -> None:
        """Test adding a speaker section with no speaker name."""
        writer = IntermediateTranscriptWriter()
        lines: list[str] = []

        writer._add_speaker_section(
            lines, None, "00:00:00.000", "00:00:05.000", ["System message"]
        )

        assert lines[0] == "### 話者不明 (00:00:00.000 - 00:00:05.000)"
        assert lines[1] == "System message"

    def test_get_statistics_basic(self) -> None:
        """Test basic statistics calculation."""
        writer = IntermediateTranscriptWriter()
        cues = self.create_sample_cues()

        stats = writer.get_statistics(cues)

        assert stats["total_cues"] == 3
        assert stats["speakers"] == ["Alice", "Bob"]  # Sorted
        assert stats["duration"] == 12.0  # end_time - start_time (12.0 - 0.0)
        assert stats["word_count"] == 124  # Count characters in all cues

    def test_get_statistics_empty(self) -> None:
        """Test statistics with empty cue list."""
        writer = IntermediateTranscriptWriter()

        stats = writer.get_statistics([])

        assert stats["total_cues"] == 0
        assert stats["speakers"] == []
        assert stats["duration"] == 0.0
        assert stats["word_count"] == 0

    def test_get_statistics_no_speakers(self) -> None:
        """Test statistics with cues that have no speakers."""
        writer = IntermediateTranscriptWriter()

        cues = [
            VTTCue(
                start_time="00:00:00.000",
                end_time="00:00:03.000",
                text="Hello world test message",
                speaker=None,
            ),
        ]

        stats = writer.get_statistics(cues)

        assert stats["total_cues"] == 1
        assert stats["speakers"] == []  # No speakers
        assert stats["duration"] == 3.0
        assert stats["word_count"] == 24  # "Hello world test message"

    def test_format_duration_basic(self) -> None:
        """Test basic duration formatting."""
        writer = IntermediateTranscriptWriter()

        # Test various durations
        assert writer.format_duration(0) == "00:00:00"
        assert writer.format_duration(30) == "00:00:30"
        assert writer.format_duration(90) == "00:01:30"
        assert writer.format_duration(3661) == "01:01:01"
        assert writer.format_duration(7200) == "02:00:00"

    def test_format_duration_fractional(self) -> None:
        """Test duration formatting with fractional seconds."""
        writer = IntermediateTranscriptWriter()

        # Fractional seconds should be truncated
        assert writer.format_duration(30.7) == "00:00:30"
        assert writer.format_duration(90.9) == "00:01:30"
        assert writer.format_duration(3661.5) == "01:01:01"

    def test_format_duration_large(self) -> None:
        """Test duration formatting with large values."""
        writer = IntermediateTranscriptWriter()

        # Large durations
        assert writer.format_duration(86400) == "24:00:00"  # 24 hours
        assert writer.format_duration(90000) == "25:00:00"  # 25 hours

    def test_speaker_grouping(self) -> None:
        """Test that consecutive cues from same speaker are grouped."""
        writer = IntermediateTranscriptWriter()

        cues = [
            VTTCue(
                start_time="00:00:00.000",
                end_time="00:00:03.000",
                text="First message",
                speaker="Alice",
            ),
            VTTCue(
                start_time="00:00:03.000",
                end_time="00:00:06.000",
                text="Second message",
                speaker="Alice",
            ),
            VTTCue(
                start_time="00:00:06.000",
                end_time="00:00:09.000",
                text="Third message",
                speaker="Bob",
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = Path(f.name)

        try:
            writer.write_markdown(cues, temp_path, "Grouped Meeting")

            content = temp_path.read_text(encoding="utf-8")

            # Alice's messages should be grouped
            assert "First message Second message" in content
            assert "### Alice (00:00:00.000 - 00:00:06.000)" in content
            assert "### Bob (00:00:06.000 - 00:00:09.000)" in content

        finally:
            temp_path.unlink()

    def test_path_string_input(self) -> None:
        """Test that string paths are handled correctly."""
        writer = IntermediateTranscriptWriter()
        cues = self.create_sample_cues()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path_str = f.name

        try:
            # Should accept string path
            writer.write_markdown(cues, temp_path_str, "String Path Test")

            # Check that file was created
            temp_path = Path(temp_path_str)
            assert temp_path.exists()

            content = temp_path.read_text(encoding="utf-8")
            assert "# String Path Test" in content

        finally:
            Path(temp_path_str).unlink()

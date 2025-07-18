"""VTT file parser for Microsoft Teams transcripts."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VTTCue:
    """Represents a single VTT cue with timing and content."""

    start_time: str
    end_time: str
    speaker: str | None
    text: str

    @property
    def start_seconds(self) -> float:
        """Convert start time to seconds."""
        return self._time_to_seconds(self.start_time)

    @property
    def end_seconds(self) -> float:
        """Convert end time to seconds."""
        return self._time_to_seconds(self.end_time)

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end_seconds - self.start_seconds

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert VTT time format to seconds.

        Args:
            time_str: Time in format "HH:MM:SS.mmm" or "MM:SS.mmm"

        Returns:
            Time in seconds as float
        """
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        else:
            raise ValueError(f"Invalid time format: {time_str}")


class VTTParser:
    """Parser for WebVTT files, specifically Microsoft Teams transcripts."""

    def __init__(self) -> None:
        """Initialize the VTT parser."""
        self._timing_pattern = self._create_timing_pattern()
        self._speaker_pattern = self._create_speaker_pattern()
        self._html_tag_pattern = self._create_html_tag_pattern()

    def _create_timing_pattern(self) -> re.Pattern[str]:
        """Create pattern for VTT cue timing line.

        Returns:
            Compiled regex pattern for timing lines
        """
        return re.compile(
            r"(\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}:\d{2}\.\d{3})"
        )

    def _create_speaker_pattern(self) -> re.Pattern[str]:
        """Create pattern for speaker voice tags.

        Returns:
            Compiled regex pattern for speaker tags
        """
        return re.compile(r"<v\s+([^>]+)>")

    def _create_html_tag_pattern(self) -> re.Pattern[str]:
        """Create pattern for removing HTML tags.

        Returns:
            Compiled regex pattern for HTML tags
        """
        return re.compile(r"<[^>]+>")

    def parse_file(self, file_path: Path) -> list[VTTCue]:
        """Parse a VTT file and return list of cues.

        Args:
            file_path: Path to the VTT file

        Returns:
            List of VTTCue objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"VTT file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content)

    def parse_content(self, content: str) -> list[VTTCue]:
        """Parse VTT content and return list of cues.

        Args:
            content: Raw VTT file content

        Returns:
            List of VTTCue objects

        Raises:
            ValueError: If content format is invalid
        """
        lines = content.strip().split("\n")

        self._validate_vtt_header(lines)

        return self._parse_vtt_lines(lines)

    def _validate_vtt_header(self, lines: list[str]) -> None:
        """Validate VTT file format.

        Args:
            lines: Lines from VTT file

        Raises:
            ValueError: If VTT header is missing or invalid
        """
        if not lines or not lines[0].strip().startswith("WEBVTT"):
            raise ValueError("Invalid VTT file: missing WEBVTT header")

    def _parse_vtt_lines(self, lines: list[str]) -> list[VTTCue]:
        """Parse VTT lines into cues.

        Args:
            lines: Lines from VTT file

        Returns:
            List of parsed VTT cues
        """
        cues: list[VTTCue] = []
        return self._process_all_lines(lines, cues)

    def _process_all_lines(self, lines: list[str], cues: list[VTTCue]) -> list[VTTCue]:
        """Process all lines starting from index 1 (skip WEBVTT header).

        Args:
            lines: Lines from VTT file
            cues: List to append parsed cues to

        Returns:
            List of parsed VTT cues
        """
        i = 1  # Skip WEBVTT header
        while i < len(lines):
            i = self._process_line(lines, i, cues)
        return cues

    def _process_line(self, lines: list[str], i: int, cues: list[VTTCue]) -> int:
        """Process a single line and update cues list.

        Args:
            lines: All lines from VTT file
            i: Current line index
            cues: List to append parsed cues to

        Returns:
            Next line index to process
        """
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith("NOTE"):
            return i + 1

        # Check for timing line
        timing_match = self._timing_pattern.match(line)
        if timing_match:
            return self._process_timing_line(lines, i, timing_match, cues)

        return i + 1

    def _process_timing_line(
        self, lines: list[str], i: int, timing_match: re.Match[str], cues: list[VTTCue]
    ) -> int:
        """Process timing line and create cue.

        Args:
            lines: All lines from VTT file
            i: Current line index
            timing_match: Regex match object for timing
            cues: List to append parsed cues to

        Returns:
            Next line index to process
        """
        start_time, end_time = self._extract_timing_info(timing_match)
        text_lines, next_i = self._collect_text_lines(lines, i + 1)

        if text_lines:
            cue = self._create_cue(start_time, end_time, text_lines)
            cues.append(cue)

        return next_i

    def _extract_timing_info(self, timing_match: re.Match[str]) -> tuple[str, str]:
        """Extract start and end times from timing match.

        Args:
            timing_match: Regex match object for timing

        Returns:
            Tuple of (start_time, end_time)
        """
        return timing_match.group(1), timing_match.group(2)

    def _collect_text_lines(
        self, lines: list[str], start_i: int
    ) -> tuple[list[str], int]:
        """Collect text lines for a cue.

        Args:
            lines: All lines from VTT file
            start_i: Starting line index

        Returns:
            Tuple of (text_lines, next_index)
        """
        text_lines: list[str] = []
        i = start_i

        while i < len(lines):
            text_line = lines[i].strip()

            if self._should_stop_collecting(text_line):
                if self._timing_pattern.match(text_line):
                    i -= 1  # Step back for next cue
                break

            text_lines.append(text_line)
            i += 1

        return text_lines, i + 1

    def _should_stop_collecting(self, text_line: str) -> bool:
        """Determine if we should stop collecting text lines.

        Args:
            text_line: Current text line

        Returns:
            True if we should stop collecting
        """
        return not text_line or bool(self._timing_pattern.match(text_line))

    def _create_cue(
        self, start_time: str, end_time: str, text_lines: list[str]
    ) -> VTTCue:
        """Create VTTCue from timing and text information.

        Args:
            start_time: Start time string
            end_time: End time string
            text_lines: List of text lines

        Returns:
            Created VTTCue object
        """
        full_text = " ".join(text_lines)
        speaker, clean_text = self._extract_speaker(full_text)

        return VTTCue(
            start_time=start_time,
            end_time=end_time,
            speaker=speaker,
            text=clean_text,
        )

    def _extract_speaker(self, text: str) -> tuple[str | None, str]:
        """Extract speaker name from VTT text.

        Args:
            text: Raw VTT text that may contain speaker tags or speaker names

        Returns:
            Tuple of (speaker_name, clean_text)
        """
        # Look for speaker voice tags first
        speaker_match = self._speaker_pattern.search(text)
        if speaker_match:
            speaker = speaker_match.group(1)
            # Remove all HTML tags
            clean_text = self._html_tag_pattern.sub("", text)
            clean_text = " ".join(clean_text.split())
            return speaker, clean_text

        # Remove HTML tags first
        clean_text = self._html_tag_pattern.sub("", text)

        # Look for Japanese speaker name pattern: "Name:" at the beginning
        speaker_name_pattern = re.compile(r"^([^\s:]+(?:\s+[^\s:]+)*)\s*:\s*(.*)$")
        speaker_name_match = speaker_name_pattern.match(clean_text.strip())

        if speaker_name_match:
            speaker = speaker_name_match.group(1).strip()
            clean_text = speaker_name_match.group(2).strip()
        else:
            speaker = None
            clean_text = " ".join(clean_text.split())

        return speaker, clean_text

    def get_speakers(self, cues: list[VTTCue]) -> list[str]:
        """Get list of unique speakers from cues.

        Args:
            cues: List of VTTCue objects

        Returns:
            List of unique speaker names
        """
        speakers: set[str] = set()
        for cue in cues:
            if cue.speaker:
                speakers.add(cue.speaker)
        return sorted(speakers)

    def get_duration(self, cues: list[VTTCue]) -> float:
        """Get total duration of the transcript.

        Args:
            cues: List of VTTCue objects

        Returns:
            Total duration in seconds
        """
        if not cues:
            return 0.0
        return cues[-1].end_seconds - cues[0].start_seconds

    def _filter_cues(
        self, cues: list[VTTCue], predicate: Callable[[VTTCue], bool]
    ) -> list[VTTCue]:
        """Generic filter function for cues.

        Args:
            cues: List of VTTCue objects
            predicate: Function that returns True for cues to include

        Returns:
            Filtered list of cues
        """
        return [cue for cue in cues if predicate(cue)]

    def filter_by_speaker(self, cues: list[VTTCue], speaker: str) -> list[VTTCue]:
        """Filter cues by speaker name.

        Args:
            cues: List of VTTCue objects
            speaker: Speaker name to filter by

        Returns:
            List of cues from the specified speaker
        """
        return self._filter_cues(cues, lambda cue: cue.speaker == speaker)

    def filter_by_time_range(
        self, cues: list[VTTCue], start_seconds: float, end_seconds: float
    ) -> list[VTTCue]:
        """Filter cues by time range.

        Args:
            cues: List of VTTCue objects
            start_seconds: Start time in seconds
            end_seconds: End time in seconds

        Returns:
            List of cues within the time range
        """

        def time_range_predicate(cue: VTTCue) -> bool:
            return cue.start_seconds >= start_seconds and cue.end_seconds <= end_seconds

        return self._filter_cues(cues, time_range_predicate)

    def filter_by_criteria(
        self,
        cues: list[VTTCue],
        speaker: str | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> list[VTTCue]:
        """Filter cues by multiple criteria.

        Args:
            cues: List of VTTCue objects
            speaker: Optional speaker name to filter by
            start_seconds: Optional start time filter
            end_seconds: Optional end time filter

        Returns:
            List of cues matching all specified criteria
        """

        def combined_predicate(cue: VTTCue) -> bool:
            if speaker is not None and cue.speaker != speaker:
                return False
            if start_seconds is not None and cue.start_seconds < start_seconds:
                return False
            if end_seconds is not None and cue.end_seconds > end_seconds:
                return False
            return True

        return self._filter_cues(cues, combined_predicate)

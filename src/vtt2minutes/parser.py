"""VTT file parser for Microsoft Teams transcripts."""

import re
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
        # Pattern for VTT cue timing line
        self._timing_pattern = re.compile(
            r"(\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}:\d{2}\.\d{3})"
        )

        # Pattern for speaker voice tags
        self._speaker_pattern = re.compile(r"<v\s+([^>]+)>")

        # Pattern for removing HTML tags
        self._html_tag_pattern = re.compile(r"<[^>]+>")

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

        # Validate VTT format
        if not lines or not lines[0].strip().startswith("WEBVTT"):
            raise ValueError("Invalid VTT file: missing WEBVTT header")

        cues = []
        i = 1  # Skip WEBVTT header

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("NOTE"):
                i += 1
                continue

            # Check for timing line
            timing_match = self._timing_pattern.match(line)
            if timing_match:
                start_time = timing_match.group(1)
                end_time = timing_match.group(2)

                # Collect text lines until next timing line or end
                text_lines = []
                i += 1

                while i < len(lines):
                    text_line = lines[i].strip()
                    if not text_line:
                        break
                    if self._timing_pattern.match(text_line):
                        # Next cue found, step back
                        i -= 1
                        break
                    text_lines.append(text_line)
                    i += 1

                if text_lines:
                    # Join text and extract speaker
                    full_text = " ".join(text_lines)
                    speaker, clean_text = self._extract_speaker(full_text)

                    cue = VTTCue(
                        start_time=start_time,
                        end_time=end_time,
                        speaker=speaker,
                        text=clean_text,
                    )
                    cues.append(cue)

            i += 1

        return cues

    def _extract_speaker(self, text: str) -> tuple[str | None, str]:
        """Extract speaker name from VTT text.

        Args:
            text: Raw VTT text that may contain speaker tags

        Returns:
            Tuple of (speaker_name, clean_text)
        """
        # Look for speaker voice tags
        speaker_match = self._speaker_pattern.search(text)
        speaker = speaker_match.group(1) if speaker_match else None

        # Remove all HTML tags
        clean_text = self._html_tag_pattern.sub("", text)

        # Clean up whitespace
        clean_text = " ".join(clean_text.split())

        return speaker, clean_text

    def get_speakers(self, cues: list[VTTCue]) -> list[str]:
        """Get list of unique speakers from cues.

        Args:
            cues: List of VTTCue objects

        Returns:
            List of unique speaker names
        """
        speakers = set()
        for cue in cues:
            if cue.speaker:
                speakers.add(cue.speaker)
        return sorted(list(speakers))

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

    def filter_by_speaker(self, cues: list[VTTCue], speaker: str) -> list[VTTCue]:
        """Filter cues by speaker name.

        Args:
            cues: List of VTTCue objects
            speaker: Speaker name to filter by

        Returns:
            List of cues from the specified speaker
        """
        return [cue for cue in cues if cue.speaker == speaker]

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
        return [
            cue
            for cue in cues
            if cue.start_seconds >= start_seconds and cue.end_seconds <= end_seconds
        ]

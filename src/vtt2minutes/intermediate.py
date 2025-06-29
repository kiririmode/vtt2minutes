"""Intermediate file output for preprocessed transcript data."""

from pathlib import Path
from typing import Any

from .parser import VTTCue


class IntermediateTranscriptWriter:
    """Writer for preprocessed transcript data in Markdown format."""

    def __init__(self) -> None:
        """Initialize the intermediate transcript writer."""
        pass

    def write_markdown(
        self,
        cues: list[VTTCue],
        output_path: Path | str,
        title: str = "前処理済み会議記録",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write preprocessed cues to a Markdown intermediate file.

        Args:
            cues: List of preprocessed VTT cues
            output_path: Path for the intermediate file
            title: Title for the transcript
            metadata: Additional metadata (date, participants, etc.)
        """
        path = Path(output_path)
        metadata = metadata or {}

        content = self._generate_markdown_content(cues, title, metadata)

        with path.open("w", encoding="utf-8") as f:
            f.write(content)

    def _generate_markdown_content(
        self,
        cues: list[VTTCue],
        title: str,
        metadata: dict[str, Any],
    ) -> str:
        """Generate Markdown content from preprocessed cues.

        Args:
            cues: List of preprocessed VTT cues
            title: Title for the transcript
            metadata: Additional metadata

        Returns:
            Formatted Markdown content
        """
        lines: list[str] = []

        # Header
        lines.append(f"# {title}")
        lines.append("")

        # Metadata
        if "date" in metadata:
            lines.append(f"**日時:** {metadata['date']}")
        if "participants" in metadata:
            participants = ", ".join(metadata["participants"])
            lines.append(f"**参加者:** {participants}")
        if "duration" in metadata:
            lines.append(f"**総時間:** {metadata['duration']}")
        if any(key in metadata for key in ["date", "participants", "duration"]):
            lines.append("")

        # Content section
        lines.append("## 発言記録")
        lines.append("")

        # Group cues by speaker for better readability
        current_speaker: str | None = object()  # type: ignore[assignment] # Unique sentinel value
        current_section: list[str] = []
        current_start_time = None

        for cue in cues:
            if cue.speaker != current_speaker:
                # Write previous section if exists
                if current_section and current_start_time is not None:
                    self._add_speaker_section(
                        lines,
                        current_speaker,
                        current_start_time,
                        cue.start_time,
                        current_section,
                    )

                # Start new section
                current_speaker = cue.speaker
                current_section = []
                current_start_time = cue.start_time

            if current_section:
                current_section.append(cue.text)
            else:
                current_section = [cue.text]

        # Write final section
        if current_section and current_start_time is not None:
            self._add_speaker_section(
                lines,
                current_speaker,
                current_start_time,
                cues[-1].end_time,
                current_section,
            )

        return "\n".join(lines)

    def _add_speaker_section(
        self,
        lines: list[str],
        speaker: str | None,
        start_time: str,
        end_time: str,
        content: list[str],
    ) -> None:
        """Add a speaker section to the markdown lines.

        Args:
            lines: List of markdown lines to append to
            speaker: Speaker name (or None for unknown)
            start_time: Start time of the section
            end_time: End time of the section
            content: List of text content for this speaker
        """
        speaker_name = speaker or "話者不明"
        section_text = " ".join(content)

        lines.append(f"### {speaker_name} ({start_time} - {end_time})")
        lines.append(section_text)
        lines.append("")

    def get_statistics(self, cues: list[VTTCue]) -> dict[str, Any]:
        """Get statistics about the preprocessed transcript.

        Args:
            cues: List of preprocessed VTT cues

        Returns:
            Dictionary with statistics
        """
        if not cues:
            return {
                "total_cues": 0,
                "speakers": [],
                "duration": 0.0,
                "word_count": 0,
            }

        speakers: set[str] = set()
        word_count = 0

        for cue in cues:
            if cue.speaker:
                speakers.add(cue.speaker)
            word_count += len(cue.text.split())

        duration = cues[-1].end_seconds - cues[0].start_seconds

        return {
            "total_cues": len(cues),
            "speakers": sorted(speakers),
            "duration": duration,
            "word_count": word_count,
        }

    def format_duration(self, seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

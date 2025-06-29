"""Meeting summarizer for creating structured minutes from VTT transcripts."""

import re
from dataclasses import dataclass
from datetime import datetime

from .parser import VTTCue


@dataclass
class MeetingSection:
    """Represents a section of the meeting minutes."""

    title: str
    content: str
    start_time: str | None = None
    end_time: str | None = None
    speaker: str | None = None


@dataclass
class ActionItem:
    """Represents an action item extracted from the meeting."""

    description: str
    assignee: str | None = None
    due_date: str | None = None
    mentioned_at: str | None = None


@dataclass
class Decision:
    """Represents a decision made during the meeting."""

    description: str
    context: str | None = None
    decided_at: str | None = None


@dataclass
class MeetingSummary:
    """Complete meeting summary with all extracted information."""

    title: str
    date: str
    duration: str
    participants: list[str]
    summary: str
    key_points: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    sections: list[MeetingSection]

    def to_markdown(self) -> str:
        """Convert the meeting summary to Markdown format.

        Returns:
            Formatted Markdown string
        """
        lines = []

        # Header
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**日時:** {self.date}")
        lines.append(f"**時間:** {self.duration}")
        lines.append(f"**参加者:** {', '.join(self.participants)}")
        lines.append("")

        # Summary
        lines.append("## 概要")
        lines.append(self.summary)
        lines.append("")

        # Key Points
        if self.key_points:
            lines.append("## 主要なポイント")
            for point in self.key_points:
                lines.append(f"- {point}")
            lines.append("")

        # Decisions
        if self.decisions:
            lines.append("## 決定事項")
            for i, decision in enumerate(self.decisions, 1):
                lines.append(f"{i}. {decision.description}")
                if decision.context:
                    lines.append(f"   - 背景: {decision.context}")
                if decision.decided_at:
                    lines.append(f"   - 決定時刻: {decision.decided_at}")
            lines.append("")

        # Action Items
        if self.action_items:
            lines.append("## アクションアイテム")
            for i, item in enumerate(self.action_items, 1):
                lines.append(f"{i}. {item.description}")
                if item.assignee:
                    lines.append(f"   - 担当者: {item.assignee}")
                if item.due_date:
                    lines.append(f"   - 期限: {item.due_date}")
                if item.mentioned_at:
                    lines.append(f"   - 言及時刻: {item.mentioned_at}")
            lines.append("")

        # Detailed sections
        if self.sections:
            lines.append("## 詳細な議事録")
            for section in self.sections:
                lines.append(f"### {section.title}")
                if section.start_time:
                    lines.append(f"**時刻:** {section.start_time}")
                    if section.end_time:
                        lines.append(f" - {section.end_time}")
                    lines.append("")
                if section.speaker:
                    lines.append(f"**発言者:** {section.speaker}")
                    lines.append("")
                lines.append(section.content)
                lines.append("")

        return "\n".join(lines)


class MeetingSummarizer:
    """Summarizer for creating structured meeting minutes from VTT cues."""

    def __init__(self) -> None:
        """Initialize the meeting summarizer."""
        # Keywords for identifying action items
        self._action_keywords = {
            # Japanese
            "宿題",
            "アクション",
            "やること",
            "課題",
            "対応",
            "確認",
            "検討",
            "調査",
            "準備",
            "作成",
            "修正",
            "更新",
            "連絡",
            "報告",
            "までに",
            "来週",
            "来月",
            "次回",
            "後で",
            "あとで",
            # English
            "action",
            "todo",
            "task",
            "homework",
            "follow up",
            "next steps",
            "by",
            "before",
            "until",
            "deadline",
            "due",
            "assign",
            "responsible",
        }

        # Keywords for identifying decisions
        self._decision_keywords = {
            # Japanese
            "決定",
            "決める",
            "採用",
            "承認",
            "合意",
            "了承",
            "確定",
            "方針",
            "方向性",
            "結論",
            "最終的に",
            "ということで",
            "に決まり",
            "で行く",
            "で進める",
            # English
            "decide",
            "decision",
            "agree",
            "agreed",
            "approve",
            "approved",
            "conclude",
            "final",
            "confirmed",
            "settled",
            "resolved",
        }

    def create_summary(
        self, cues: list[VTTCue], title: str | None = None
    ) -> MeetingSummary:
        """Create a comprehensive meeting summary from VTT cues.

        Args:
            cues: List of preprocessed VTT cues
            title: Optional meeting title

        Returns:
            Complete meeting summary
        """
        if not cues:
            return self._create_empty_summary(title or "Empty Meeting")

        # Extract basic information
        participants = self._extract_participants(cues)
        duration = self._format_duration(cues[-1].end_seconds - cues[0].start_seconds)
        date = datetime.now().strftime("%Y年%m月%d日")

        # Generate summary and key points
        summary = self._generate_summary(cues)
        key_points = self._extract_key_points(cues)

        # Extract structured information
        decisions = self._extract_decisions(cues)
        action_items = self._extract_action_items(cues)
        sections = self._create_sections(cues)

        return MeetingSummary(
            title=title or f"会議 - {date}",
            date=date,
            duration=duration,
            participants=participants,
            summary=summary,
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            sections=sections,
        )

    def _create_empty_summary(self, title: str) -> MeetingSummary:
        """Create an empty meeting summary.

        Args:
            title: Meeting title

        Returns:
            Empty meeting summary
        """
        return MeetingSummary(
            title=title,
            date=datetime.now().strftime("%Y年%m月%d日"),
            duration="00:00:00",
            participants=[],
            summary="会議の記録がありません。",
            key_points=[],
            decisions=[],
            action_items=[],
            sections=[],
        )

    def _extract_participants(self, cues: list[VTTCue]) -> list[str]:
        """Extract list of meeting participants.

        Args:
            cues: List of VTT cues

        Returns:
            List of participant names
        """
        speakers: set[str] = set()
        for cue in cues:
            if cue.speaker:
                speakers.add(cue.speaker)

        return sorted(list(speakers))

    def _format_duration(self, seconds: float) -> str:
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

    def _generate_summary(self, cues: list[VTTCue]) -> str:
        """Generate a brief summary of the meeting.

        Args:
            cues: List of VTT cues

        Returns:
            Meeting summary text
        """
        # Simple summary generation - in a real implementation,
        # this could use NLP techniques or AI for better summarization

        total_words = sum(len(cue.text.split()) for cue in cues)
        speaker_count = len(set(cue.speaker for cue in cues if cue.speaker))

        # Extract first few significant statements
        significant_statements = []
        for cue in cues[:5]:  # Look at first 5 cues
            if len(cue.text) > 20:  # Only include substantial text
                significant_statements.append(cue.text[:100] + "...")

        summary_parts = [
            f"この会議には{speaker_count}名が参加し、"
            f"合計{total_words}語の発言が記録されました。"
        ]

        if significant_statements:
            summary_parts.append("主な議論内容:")
            summary_parts.extend(f"- {stmt}" for stmt in significant_statements)

        return " ".join(summary_parts)

    def _extract_key_points(self, cues: list[VTTCue]) -> list[str]:
        """Extract key points from the meeting.

        Args:
            cues: List of VTT cues

        Returns:
            List of key points
        """
        key_points = []

        # Look for important statements (longer texts, questions, decisions)
        for cue in cues:
            text = cue.text.strip()

            # Skip very short texts
            if len(text) < 30:
                continue

            # Look for questions
            if "?" in text or "？" in text:
                key_points.append(f"質問: {text}")

            # Look for important statements (contains keywords)
            important_keywords = {
                "重要",
                "大切",
                "ポイント",
                "課題",
                "問題",
                "解決",
                "必要",
                "should",
                "must",
                "important",
                "critical",
            }

            if any(keyword in text.lower() for keyword in important_keywords):
                key_points.append(text)

            # Limit number of key points
            if len(key_points) >= 10:
                break

        return key_points

    def _extract_decisions(self, cues: list[VTTCue]) -> list[Decision]:
        """Extract decisions made during the meeting.

        Args:
            cues: List of VTT cues

        Returns:
            List of decisions
        """
        decisions = []

        for cue in cues:
            text = cue.text.lower()

            # Check if text contains decision keywords
            if any(keyword in text for keyword in self._decision_keywords):
                decision = Decision(
                    description=cue.text,
                    decided_at=cue.start_time,
                    context=None,  # Could be enhanced to include surrounding context
                )
                decisions.append(decision)

        return decisions

    def _extract_action_items(self, cues: list[VTTCue]) -> list[ActionItem]:
        """Extract action items from the meeting.

        Args:
            cues: List of VTT cues

        Returns:
            List of action items
        """
        action_items = []

        for cue in cues:
            text = cue.text.lower()

            # Check if text contains action keywords
            if any(keyword in text for keyword in self._action_keywords):
                # Try to extract assignee from the text
                assignee = cue.speaker

                # Try to extract due date
                due_date = self._extract_due_date(cue.text)

                action_item = ActionItem(
                    description=cue.text,
                    assignee=assignee,
                    due_date=due_date,
                    mentioned_at=cue.start_time,
                )
                action_items.append(action_item)

        return action_items

    def _extract_due_date(self, text: str) -> str | None:
        """Extract due date from text if present.

        Args:
            text: Text to search for due dates

        Returns:
            Due date string if found, None otherwise
        """
        # Simple pattern matching for common date expressions
        date_patterns = [
            r"来週",
            r"来月",
            r"次回",
            r"明日",
            r"今週中",
            r"今月中",
            r"\d+月\d+日",
            r"\d{4}/\d{1,2}/\d{1,2}",
            r"next week",
            r"next month",
            r"tomorrow",
            r"by \w+day",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _create_sections(self, cues: list[VTTCue]) -> list[MeetingSection]:
        """Create structured sections from the meeting transcript.

        Args:
            cues: List of VTT cues

        Returns:
            List of meeting sections
        """
        sections = []
        current_section = []
        current_speaker = None
        section_start = None

        for cue in cues:
            # Start new section when speaker changes
            if cue.speaker != current_speaker:
                # Save previous section if it exists
                if current_section and current_speaker:
                    content = " ".join(c.text for c in current_section)
                    section = MeetingSection(
                        title=f"{current_speaker}の発言",
                        content=content,
                        start_time=section_start,
                        end_time=current_section[-1].end_time,
                        speaker=current_speaker,
                    )
                    sections.append(section)

                # Start new section
                current_section = [cue]
                current_speaker = cue.speaker
                section_start = cue.start_time
            else:
                current_section.append(cue)

        # Add final section
        if current_section and current_speaker:
            content = " ".join(c.text for c in current_section)
            section = MeetingSection(
                title=f"{current_speaker}の発言",
                content=content,
                start_time=section_start,
                end_time=current_section[-1].end_time,
                speaker=current_speaker,
            )
            sections.append(section)

        return sections

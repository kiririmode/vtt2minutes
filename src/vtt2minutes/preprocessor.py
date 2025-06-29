"""Text preprocessing for VTT transcripts to improve quality."""

import re
from dataclasses import dataclass
from typing import Any

from .parser import VTTCue


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    # Filler words to remove (common Japanese/English filler words)
    filler_words: set[str] | None = None

    # Minimum text length to keep (characters)
    min_text_length: int = 3

    # Maximum duplicate similarity threshold (0.0-1.0)
    duplicate_threshold: float = 0.8

    # Minimum duration for a cue to be considered valid (seconds)
    min_duration: float = 0.5

    # Maximum duration gap to merge consecutive cues (seconds)
    merge_gap_threshold: float = 2.0

    # Whether to merge consecutive cues from the same speaker
    merge_same_speaker: bool = True

    # Whether to fix common transcription errors
    fix_transcription_errors: bool = True

    def __post_init__(self) -> None:
        """Initialize default filler words if not provided."""
        if self.filler_words is None:
            self.filler_words = {
                # Japanese filler words
                "えー",
                "あー",
                "うー",
                "そのー",
                "なんか",
                "まあ",
                "ちょっと",
                "えっと",
                "あのー",
                "そうですね",
                "はい",
                "ええ",
                "うん",
                # English filler words
                "um",
                "uh",
                "like",
                "you know",
                "well",
                "okay",
                "right",
                "actually",
                "basically",
                "literally",
                "honestly",
                # Common transcription artifacts
                "[音声が途切れました]",
                "[雑音]",
                "[不明瞭]",
                "[咳]",
                "[笑い]",
            }


class TextPreprocessor:
    """Preprocessor for cleaning and improving VTT transcript text."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Initialize the text preprocessor.

        Args:
            config: Configuration for preprocessing, defaults to PreprocessingConfig()
        """
        self.config = config or PreprocessingConfig()

        # Common transcription error patterns
        self._error_patterns = {
            # Repeated characters
            r"(.)\1{3,}": r"\1",
            # Multiple spaces
            r"\s+": " ",
            # Repeated words
            r"\b(\w+)\s+\1\b": r"\1",
            # Common OCR/ASR errors (Japanese context)
            r"です。です。": "です。",
            r"ます。ます。": "ます。",
            r"ています。ています。": "ています。",
        }

    def preprocess_cues(self, cues: list[VTTCue]) -> list[VTTCue]:
        """Preprocess a list of VTT cues.

        Args:
            cues: List of VTTCue objects to preprocess

        Returns:
            List of preprocessed VTTCue objects
        """
        if not cues:
            return cues

        # Step 1: Clean individual cues
        cleaned_cues = [self._clean_cue(cue) for cue in cues]

        # Step 2: Filter out invalid cues
        valid_cues = self._filter_invalid_cues(cleaned_cues)

        # Step 3: Remove duplicates
        deduplicated_cues = self._remove_duplicates(valid_cues)

        # Step 4: Merge consecutive cues if configured
        if self.config.merge_same_speaker:
            merged_cues = self._merge_consecutive_cues(deduplicated_cues)
        else:
            merged_cues = deduplicated_cues

        # Step 5: Final cleanup
        final_cues = [self._final_cleanup(cue) for cue in merged_cues]

        return [cue for cue in final_cues if cue.text.strip()]

    def _clean_cue(self, cue: VTTCue) -> VTTCue:
        """Clean a single VTT cue.

        Args:
            cue: VTTCue to clean

        Returns:
            Cleaned VTTCue
        """
        text = cue.text

        # Remove filler words
        text = self._remove_filler_words(text)

        # Fix transcription errors
        if self.config.fix_transcription_errors:
            text = self._fix_transcription_errors(text)

        # Normalize whitespace
        text = " ".join(text.split())

        return VTTCue(
            start_time=cue.start_time,
            end_time=cue.end_time,
            speaker=cue.speaker,
            text=text,
        )

    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words from text.

        Args:
            text: Input text

        Returns:
            Text with filler words removed
        """
        if not self.config.filler_words:
            return text

        # Split by whitespace and punctuation to get individual words
        words = re.findall(r"[^\s、。，．！？!?]+", text)
        filtered_words = []

        for word in words:
            # Remove punctuation for comparison
            clean_word = re.sub(r"[^\w]", "", word.lower())
            if clean_word not in self.config.filler_words:
                filtered_words.append(word)

        # Reconstruct text preserving some punctuation
        result = " ".join(filtered_words)

        # Add back sentence-ending punctuation if original had it
        if text.endswith(("。", ".", "！", "!", "？", "?")):
            if not result.endswith(("。", ".", "！", "!", "？", "?")):
                # Preserve original punctuation type when possible
                if text.endswith("。"):
                    result += "。"
                elif text.endswith("."):
                    result += "."
                elif re.search(r"[ひらがなカタカナ漢字]", result):
                    result += "。"
                else:
                    result += "."
        elif text.strip():  # If original had no ending punctuation but text exists
            if not result.endswith(("。", ".", "！", "!", "？", "?")):
                if re.search(r"[ひらがなカタカナ漢字]", result):
                    result += "。"
                else:
                    result += "."

        return result

    def _fix_transcription_errors(self, text: str) -> str:
        """Fix common transcription errors.

        Args:
            text: Input text

        Returns:
            Text with errors fixed
        """
        for pattern, replacement in self._error_patterns.items():
            text = re.sub(pattern, replacement, text)

        return text

    def _filter_invalid_cues(self, cues: list[VTTCue]) -> list[VTTCue]:
        """Filter out invalid cues based on configuration.

        Args:
            cues: List of VTTCue objects

        Returns:
            List of valid cues
        """
        valid_cues = []

        for cue in cues:
            # Check minimum text length
            if len(cue.text.strip()) < self.config.min_text_length:
                continue

            # Check minimum duration
            if cue.duration < self.config.min_duration:
                continue

            # Check for empty or meaningless text
            if not cue.text.strip() or cue.text.strip() in ["。", "、", ".", ","]:
                continue

            valid_cues.append(cue)

        return valid_cues

    def _remove_duplicates(self, cues: list[VTTCue]) -> list[VTTCue]:
        """Remove duplicate or very similar cues.

        Args:
            cues: List of VTTCue objects

        Returns:
            List of cues with duplicates removed
        """
        if not cues:
            return cues

        unique_cues = [cues[0]]  # Always keep the first cue

        for cue in cues[1:]:
            is_duplicate = False

            for existing_cue in unique_cues:
                if self._are_similar(cue.text, existing_cue.text):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_cues.append(cue)

        return unique_cues

    def _are_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar enough to be considered duplicates.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts are similar, False otherwise
        """
        # Simple similarity check based on word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union)
        return similarity >= self.config.duplicate_threshold

    def _merge_consecutive_cues(self, cues: list[VTTCue]) -> list[VTTCue]:
        """Merge consecutive cues from the same speaker.

        Args:
            cues: List of VTTCue objects

        Returns:
            List of cues with consecutive ones merged
        """
        if len(cues) <= 1:
            return cues

        merged_cues = []
        current_cue = cues[0]

        for next_cue in cues[1:]:
            # Check if cues can be merged
            if (
                current_cue.speaker == next_cue.speaker
                and current_cue.speaker is not None
                and next_cue.start_seconds - current_cue.end_seconds
                <= self.config.merge_gap_threshold
            ):
                # Merge the cues
                merged_text = f"{current_cue.text} {next_cue.text}".strip()
                current_cue = VTTCue(
                    start_time=current_cue.start_time,
                    end_time=next_cue.end_time,
                    speaker=current_cue.speaker,
                    text=merged_text,
                )
            else:
                # Can't merge, add current cue and move to next
                merged_cues.append(current_cue)
                current_cue = next_cue

        # Add the last cue
        merged_cues.append(current_cue)

        return merged_cues

    def _final_cleanup(self, cue: VTTCue) -> VTTCue:
        """Perform final cleanup on a cue.

        Args:
            cue: VTTCue to clean up

        Returns:
            Final cleaned VTTCue
        """
        text = cue.text

        # Remove extra punctuation
        text = re.sub(r"[。]{2,}", "。", text)
        text = re.sub(r"[、]{2,}", "、", text)
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[,]{2,}", ",", text)

        # Ensure proper sentence endings
        text = text.strip()
        if text and not text.endswith((".", "。", "!", "?", "！", "？")):
            # Add appropriate punctuation based on language
            if re.search(r"[ひらがなカタカナ漢字]", text):
                text += "。"
            else:
                text += "."

        return VTTCue(
            start_time=cue.start_time,
            end_time=cue.end_time,
            speaker=cue.speaker,
            text=text,
        )

    def get_statistics(
        self, original_cues: list[VTTCue], processed_cues: list[VTTCue]
    ) -> dict[str, Any]:
        """Get preprocessing statistics.

        Args:
            original_cues: Original list of cues
            processed_cues: Processed list of cues

        Returns:
            Dictionary with preprocessing statistics
        """
        return {
            "original_count": len(original_cues),
            "processed_count": len(processed_cues),
            "removed_count": len(original_cues) - len(processed_cues),
            "removal_rate": (len(original_cues) - len(processed_cues))
            / len(original_cues)
            if original_cues
            else 0,
            "original_duration": sum(cue.duration for cue in original_cues),
            "processed_duration": sum(cue.duration for cue in processed_cues),
            "original_text_length": sum(len(cue.text) for cue in original_cues),
            "processed_text_length": sum(len(cue.text) for cue in processed_cues),
        }

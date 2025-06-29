"""Text preprocessing for VTT transcripts to improve quality."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parser import VTTCue


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    # Filler words to remove (common Japanese/English filler words)
    filler_words: set[str] | None = None

    # Path to filler words file (overrides default filler_words if provided)
    filler_words_file: Path | str | None = None

    # Word replacement rules (original -> replacement)
    replacement_rules: dict[str, str] | None = None

    # Path to replacement rules file (overrides default replacement_rules if provided)
    replacement_rules_file: Path | str | None = None

    # Whether to enable word replacement
    enable_word_replacement: bool = True

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
        """Initialize filler words and replacement rules from file or defaults."""
        # If a filler words file is specified, load from file
        if self.filler_words_file is not None:
            self.filler_words = self._load_filler_words_from_file(
                self.filler_words_file
            )
        elif self.filler_words is None:
            # Use default filler words if none provided
            self.filler_words = self._get_default_filler_words()

        # If a replacement rules file is specified, load from file
        if self.replacement_rules_file is not None:
            self.replacement_rules = self._load_replacement_rules_from_file(
                self.replacement_rules_file
            )
        elif self.replacement_rules is None:
            # Use empty dict if none provided
            self.replacement_rules = {}

    def _load_filler_words_from_file(self, file_path: Path | str) -> set[str]:
        """Load filler words from a text file.

        Args:
            file_path: Path to the filler words file

        Returns:
            Set of filler words loaded from the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be read
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Filler words file not found: {path}")

        try:
            filler_words: set[str] = set()
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        filler_words.add(line)
            return filler_words
        except Exception as e:
            raise ValueError(f"Failed to read filler words file {path}: {e}") from e

    def _get_default_filler_words(self) -> set[str]:
        """Get the default set of filler words.

        Returns:
            Default set of filler words
        """
        return {
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

    def _load_replacement_rules_from_file(
        self, file_path: Path | str
    ) -> dict[str, str]:
        """Load replacement rules from a text file.

        Args:
            file_path: Path to the replacement rules file

        Returns:
            Dictionary of replacement rules (original -> replacement)

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Replacement rules file not found: {file_path}")

        replacement_rules: dict[str, str] = {}
        try:
            with file_path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse "original -> replacement" format
                    if " -> " in line:
                        original, replacement = line.split(" -> ", 1)
                        original = original.strip()
                        replacement = replacement.strip()
                        if original and replacement:
                            replacement_rules[original] = replacement
                    else:
                        # Log warning for invalid format but continue processing
                        print(
                            f"Warning: Invalid format at line {line_num} in "
                            f"{file_path}: {line}"
                        )
        except Exception as e:
            raise ValueError(
                f"Error reading replacement rules file {file_path}: {e}"
            ) from e

        return replacement_rules


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

        # Apply word replacement if enabled
        if self.config.enable_word_replacement:
            text = self._apply_word_replacement(text)

        # Remove filler words
        text = self._remove_filler_words(text)

        # Fix transcription errors
        if self.config.fix_transcription_errors:
            text = self._fix_transcription_errors(text)

        # Normalize whitespace appropriately for the language
        if re.search(r"[ひらがなカタカナ漢字]", text):
            # For Japanese text, normalize spaces more carefully
            # Remove multiple consecutive spaces but preserve natural breaks
            text = re.sub(r"\s{2,}", " ", text)
            # Remove spaces around Japanese punctuation
            text = re.sub(r"\s*([。、！？])\s*", r"\1", text)
            text = text.strip()
        else:
            # For non-Japanese text, normalize to single spaces
            text = " ".join(text.split())

        return VTTCue(
            start_time=cue.start_time,
            end_time=cue.end_time,
            speaker=cue.speaker,
            text=text,
        )

    def _apply_word_replacement(self, text: str) -> str:
        """Apply word replacement rules to text.

        Args:
            text: Input text

        Returns:
            Text with words replaced according to replacement rules
        """
        if not self.config.enable_word_replacement or not self.config.replacement_rules:
            return text

        # Sort by length (longest first) to prevent partial replacement
        sorted_rules = sorted(
            self.config.replacement_rules.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        result_text = text
        for original, replacement in sorted_rules:
            # For Japanese text, we do simple string replacement (no word boundaries)
            # since Japanese doesn't have clear word boundaries like English
            result_text = result_text.replace(original, replacement)

        return result_text

    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words from text.

        Args:
            text: Input text

        Returns:
            Text with filler words removed
        """
        if not self.config.filler_words:
            return text

        # Check if text is primarily Japanese
        is_japanese = bool(re.search(r"[ひらがなカタカナ漢字]", text))

        # For Japanese text, split preserving punctuation structure
        if is_japanese:
            # Split into segments at punctuation marks, preserving them
            segments = re.split(r"([、。，．！？])", text)
            result_segments: list[str] = []

            for segment in segments:
                if segment in ["、", "。", "，", "．", "！", "？"]:
                    # Keep punctuation as-is
                    result_segments.append(segment)
                else:
                    # Process words in this segment
                    words = segment.split()
                    filtered_words: list[str] = []

                    for word in words:
                        # Remove punctuation for comparison
                        clean_word = re.sub(r"[^\w]", "", word.lower())
                        if clean_word not in self.config.filler_words:
                            filtered_words.append(word)

                    # Join filtered words without extra spaces
                    if filtered_words:
                        result_segments.append("".join(filtered_words))

            result = "".join(result_segments)
            # Clean up leading punctuation
            result = re.sub(r"^[、，]", "", result)
        else:
            # For English/other languages, process normally
            words = re.findall(r"[^\s、。，．！？!?]+", text)
            filtered_words: list[str] = []

            for word in words:
                # Remove punctuation for comparison
                clean_word = re.sub(r"[^\w]", "", word.lower())
                if clean_word not in self.config.filler_words:
                    filtered_words.append(word)

            result = " ".join(filtered_words)

        # Add back sentence-ending punctuation if original had it and result doesn't
        if text.endswith(("。", ".", "！", "!", "？", "?")):
            if not result.endswith(("。", ".", "！", "!", "？", "?")):
                # Preserve original punctuation type when possible
                if text.endswith("。"):
                    result += "。"
                elif text.endswith("."):
                    result += "."
                elif is_japanese:
                    result += "。"
                else:
                    result += "."
        elif text.strip() and not result.endswith(("。", ".", "！", "!", "？", "?")):
            # Add appropriate punctuation if none exists
            if is_japanese:
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
        valid_cues: list[VTTCue] = []

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

        merged_cues: list[VTTCue] = []
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

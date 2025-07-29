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
        self._error_patterns = self._create_error_patterns()

    def _create_error_patterns(self) -> dict[str, str]:
        """Create transcription error patterns.

        Returns:
            Dictionary mapping error patterns to corrections
        """
        return {
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

    def _clean_cue(self, cue: VTTCue, final_cleanup: bool = False) -> VTTCue:
        """Clean a single VTT cue.

        Args:
            cue: VTTCue to clean
            final_cleanup: Whether this is final cleanup (limited processing)

        Returns:
            Cleaned VTTCue
        """
        if final_cleanup:
            return self._apply_final_cleanup(cue)
        else:
            return self._apply_full_cleanup(cue)

    def _apply_final_cleanup(self, cue: VTTCue) -> VTTCue:
        """Apply final cleanup processing to a cue."""
        text = cue.text
        text = self._remove_extra_punctuation(text)
        text = self._ensure_proper_endings(text)
        text = self._normalize_whitespace(text)
        return self._create_cleaned_cue(cue, text)

    def _apply_full_cleanup(self, cue: VTTCue) -> VTTCue:
        """Apply full cleanup processing to a cue."""
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
        text = self._normalize_whitespace(text)

        return self._create_cleaned_cue(cue, text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace appropriately for the language.

        Args:
            text: Text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Check if text contains Japanese characters
        has_japanese = re.search(r"[ひらがなカタカナ漢字]", text) is not None

        if has_japanese:
            return self._normalize_japanese_whitespace(text)
        else:
            return self._normalize_standard_whitespace(text)

    def _normalize_japanese_whitespace(self, text: str) -> str:
        """Normalize whitespace for Japanese text."""
        # Remove multiple consecutive spaces but preserve natural breaks
        text = re.sub(r"\s{2,}", " ", text)
        # Remove spaces around Japanese punctuation
        text = re.sub(r"\s*([。、！？])\s*", r"\1", text)
        return text.strip()

    def _normalize_standard_whitespace(self, text: str) -> str:
        """Normalize whitespace for non-Japanese text."""
        return " ".join(text.split())

    def _ensure_proper_endings(self, text: str) -> str:
        """Ensure proper sentence endings.

        Args:
            text: Text to process

        Returns:
            Text with proper sentence endings
        """
        text = text.strip()
        if text and not text.endswith((".", "。", "!", "?", "！", "？")):
            # Add appropriate punctuation based on language
            if re.search(r"[ひらがなカタカナ漢字]", text):
                text += "。"
            else:
                text += "."
        return text

    def _remove_extra_punctuation(self, text: str) -> str:
        """Remove extra punctuation marks.

        Args:
            text: Text to process

        Returns:
            Text with extra punctuation removed
        """
        text = re.sub(r"[。]{2,}", "。", text)
        text = re.sub(r"[、]{2,}", "、", text)
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[,]{2,}", ",", text)
        return text

    def _create_cleaned_cue(self, original_cue: VTTCue, cleaned_text: str) -> VTTCue:
        """Create a new VTTCue with cleaned text.

        Args:
            original_cue: Original VTTCue
            cleaned_text: Cleaned text

        Returns:
            New VTTCue with cleaned text
        """
        return VTTCue(
            start_time=original_cue.start_time,
            end_time=original_cue.end_time,
            speaker=original_cue.speaker,
            text=cleaned_text,
        )

    def _apply_word_replacement(self, text: str) -> str:
        """Apply word replacement rules to text.

        Args:
            text: Input text

        Returns:
            Text with words replaced according to replacement rules
        """
        if not self._should_apply_word_replacement():
            return text

        if self.config.replacement_rules:
            return self._perform_text_replacements(text, self.config.replacement_rules)
        return text

    def _should_apply_word_replacement(self) -> bool:
        """Check if word replacement should be applied.

        Returns:
            True if word replacement should be applied
        """
        return (
            self.config.enable_word_replacement
            and self.config.replacement_rules is not None
        )

    def _perform_text_replacements(self, text: str, rules: dict[str, str]) -> str:
        """Perform text replacements using given rules.

        Args:
            text: Text to process
            rules: Replacement rules dictionary

        Returns:
            Text with replacements applied
        """
        # Sort by length (longest first) to prevent partial replacement
        sorted_rules = sorted(rules.items(), key=lambda x: len(x[0]), reverse=True)

        result_text = text
        for original, replacement in sorted_rules:
            # For Japanese text, we do simple string replacement (no word boundaries)
            # since Japanese doesn't have clear word boundaries like English
            result_text = result_text.replace(original, replacement)

        return result_text

    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words from Japanese text.

        Args:
            text: Input Japanese text

        Returns:
            Text with filler words removed
        """
        if not self.config.filler_words:
            return text

        # Split into segments and process each
        segments = re.split(r"([、。，．！？])", text)
        result_segments = self._process_segments(segments)
        result = "".join(result_segments)

        # Clean up and add punctuation
        result = self._cleanup_and_add_punctuation(result, text)
        return result

    def _process_segments(self, segments: list[str]) -> list[str]:
        """Process text segments to remove filler words.

        Args:
            segments: List of text segments split by punctuation

        Returns:
            List of processed segments
        """
        result_segments: list[str] = []
        punctuation_marks = {"、", "。", "，", "．", "！", "？"}

        for segment in segments:
            if segment in punctuation_marks:
                result_segments.append(segment)
            else:
                filtered_segment = self._filter_words_in_segment(segment)
                if filtered_segment:
                    result_segments.append(filtered_segment)

        return result_segments

    def _filter_words_in_segment(self, segment: str) -> str:
        """Filter filler words from a text segment.

        Args:
            segment: Text segment to process

        Returns:
            Segment with filler words removed
        """
        if not self.config.filler_words:
            return segment

        words = segment.split()
        filtered_words: list[str] = []

        for word in words:
            if not self._is_filler_word(word):
                filtered_words.append(word)

        return " ".join(filtered_words)

    def _is_filler_word(self, word: str) -> bool:
        """Check if a word is a filler word."""
        clean_word = re.sub(r"[^\w]", "", word.lower())
        filler_words = self.config.filler_words
        return filler_words is not None and clean_word in filler_words

    def _cleanup_and_add_punctuation(self, result: str, original_text: str) -> str:
        """Clean up result and add appropriate punctuation.

        Args:
            result: Processed text
            original_text: Original input text

        Returns:
            Cleaned text with proper punctuation
        """
        # Clean up leading punctuation
        result = re.sub(r"^[、，]", "", result)

        return self._add_appropriate_punctuation(result, original_text)

    def _add_appropriate_punctuation(self, text: str, original_text: str) -> str:
        """Add appropriate punctuation to text."""
        ending_punctuation = ("。", ".", "！", "!", "？", "?")

        if original_text.endswith(ending_punctuation):
            if not text.endswith(ending_punctuation):
                text += self._get_appropriate_ending(original_text)
        elif original_text.strip() and not text.endswith(ending_punctuation):
            text += "。"

        return text

    def _get_appropriate_ending(self, text: str) -> str:
        """Get appropriate sentence ending punctuation.

        Args:
            text: Original text to check ending

        Returns:
            Appropriate punctuation mark
        """
        if text.endswith("。"):
            return "。"
        elif text.endswith("."):
            return "."
        else:
            return "。"

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
        return [cue for cue in cues if self._is_valid_cue(cue)]

    def _is_valid_cue(self, cue: VTTCue) -> bool:
        """Check if a cue is valid based on configuration.

        Args:
            cue: VTTCue to validate

        Returns:
            True if cue is valid
        """
        return (
            self._has_valid_text_length(cue)
            and self._has_valid_duration(cue)
            and self._has_meaningful_content(cue)
        )

    def _check_cue_property(self, cue: VTTCue, check_type: str) -> bool:
        """Generic cue property checker.

        Args:
            cue: VTTCue to check
            check_type: Type of check ('text_length', 'duration', 'meaningful')

        Returns:
            True if property check passes
        """
        if check_type == "text_length":
            return len(cue.text.strip()) >= self.config.min_text_length
        elif check_type == "duration":
            return cue.duration >= self.config.min_duration
        elif check_type == "meaningful":
            meaningless_texts = {"。", "、", ".", ","}
            stripped_text = cue.text.strip()
            return bool(stripped_text and stripped_text not in meaningless_texts)
        return False

    def _has_valid_text_length(self, cue: VTTCue) -> bool:
        """Check if cue has valid text length."""
        return self._check_cue_property(cue, "text_length")

    def _has_valid_duration(self, cue: VTTCue) -> bool:
        """Check if cue has valid duration."""
        return self._check_cue_property(cue, "duration")

    def _has_meaningful_content(self, cue: VTTCue) -> bool:
        """Check if cue has meaningful content."""
        return self._check_cue_property(cue, "meaningful")

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
            if self._can_merge_cues(current_cue, next_cue):
                current_cue = self._merge_two_cues(current_cue, next_cue)
            else:
                merged_cues.append(current_cue)
                current_cue = next_cue

        merged_cues.append(current_cue)
        return merged_cues

    def _can_merge_cues(self, current: VTTCue, next_cue: VTTCue) -> bool:
        """Check if two cues can be merged.

        Args:
            current: Current cue
            next_cue: Next cue to potentially merge

        Returns:
            True if cues can be merged
        """
        return (
            current.speaker == next_cue.speaker
            and current.speaker is not None
            and next_cue.start_seconds - current.end_seconds
            <= self.config.merge_gap_threshold
        )

    def _merge_two_cues(self, current: VTTCue, next_cue: VTTCue) -> VTTCue:
        """Merge two cues into one.

        Args:
            current: Current cue
            next_cue: Next cue to merge

        Returns:
            Merged cue
        """
        merged_text = f"{current.text} {next_cue.text}".strip()
        return VTTCue(
            start_time=current.start_time,
            end_time=next_cue.end_time,
            speaker=current.speaker,
            text=merged_text,
        )

    def _final_cleanup(self, cue: VTTCue) -> VTTCue:
        """Perform final cleanup on a cue.

        Args:
            cue: VTTCue to clean up

        Returns:
            Final cleaned VTTCue
        """
        return self._clean_cue(cue, final_cleanup=True)

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

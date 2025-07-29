"""Single file processing functionality abstracted from CLI main function."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .bedrock import BedrockError, BedrockMeetingMinutesGenerator
from .intermediate import IntermediateTranscriptWriter
from .parser import VTTCue, VTTParser
from .preprocessor import PreprocessingConfig, TextPreprocessor

T = TypeVar("T")


@dataclass
class ProcessingConfig:
    """Configuration for VTT file processing."""

    # Preprocessing options
    no_preprocessing: bool = False
    min_duration: float = 0.5
    merge_threshold: float = 2.0
    duplicate_threshold: float = 0.8
    filter_words_file: Path | None = None
    replacement_rules_file: Path | None = None

    # Bedrock options
    bedrock_model: str | None = None
    bedrock_inference_profile_id: str | None = None
    bedrock_region: str = "ap-northeast-1"
    prompt_template: Path | None = None

    # Output options
    overwrite: bool = False
    verbose: bool = False
    stats: bool = False
    delete_vtt_file: bool = False


@dataclass
class ProcessingResult:
    """Result of processing a single VTT file."""

    success: bool
    input_file: Path
    output_file: Path | None = None
    intermediate_file: Path | None = None
    error: str | None = None
    statistics: dict[str, Any] | None = None


class VTTFileProcessor:
    """Processes a single VTT file to generate meeting minutes."""

    def __init__(self, config: ProcessingConfig) -> None:
        """Initialize the processor with configuration.

        Args:
            config: Processing configuration
        """
        self.config = config
        self.console = Console()

    def _execute_with_progress(
        self,
        operation: Callable[[], T],
        task_description: str,
        success_description: str,
        error_message: str,
        progress: Progress,
    ) -> T:
        """Execute an operation with progress tracking and error handling.

        Args:
            operation: Function to execute
            task_description: Description to show while executing
            success_description: Description to show on success
            error_message: Error message prefix for failures
            progress: Progress object to track task

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If operation fails
        """
        task = progress.add_task(task_description, total=None)

        try:
            result = operation()
            progress.update(task, description=success_description)
            return result
        except Exception as e:
            progress.stop()
            raise RuntimeError(f"{error_message}: {e}") from e

    def _create_bedrock_generator(self) -> BedrockMeetingMinutesGenerator:
        """Create a BedrockMeetingMinutesGenerator with current config."""
        return BedrockMeetingMinutesGenerator(
            region_name=self.config.bedrock_region,
            model_id=self.config.bedrock_model,
            inference_profile_id=self.config.bedrock_inference_profile_id,
            prompt_template_file=self.config.prompt_template,
        )

    def _check_file_overwrite(self, file_path: Path, file_type: str) -> None:
        """Check if file exists and handle overwrite logic.

        Args:
            file_path: Path to check
            file_type: Type of file for error message

        Raises:
            FileExistsError: If file exists and overwrite is disabled
        """
        if file_path.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"{file_type} already exists: {file_path}. "
                "Use --overwrite to overwrite existing files."
            )

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        title: str | None = None,
        intermediate_file: Path | None = None,
        chat_prompt_file: Path | None = None,
    ) -> ProcessingResult:
        """Process a single VTT file.

        Args:
            input_file: Path to input VTT file
            output_file: Path for output markdown file
            title: Meeting title (optional)
            intermediate_file: Path for intermediate file (optional)
            chat_prompt_file: If provided, generates chat prompt instead of Bedrock

        Returns:
            ProcessingResult with outcome and details
        """
        try:
            # Initialize components
            parser, preprocessor = self._initialize_components()

            # Process with progress indication
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                # Step 1: Parse VTT file
                cues = self._parse_vtt_file(parser, input_file, progress)

                # Step 2: Preprocess (if enabled)
                original_cues = cues.copy()
                cues = self._preprocess_cues(preprocessor, cues, progress)

                # Step 3: Save intermediate file
                intermediate_path = self._save_intermediate_file(
                    cues, output_file, intermediate_file, title, progress
                )

                # Step 4: Generate chat prompt file if requested
                if chat_prompt_file:
                    self._generate_chat_prompt(
                        intermediate_path, chat_prompt_file, title, progress
                    )
                    
                    # Delete VTT file if requested and processing was successful
                    if self.config.delete_vtt_file:
                        self._delete_vtt_file(input_file)
                    
                    return ProcessingResult(
                        success=True,
                        input_file=input_file,
                        output_file=chat_prompt_file,
                        intermediate_file=intermediate_path,
                    )

                # Step 5: Generate AI-powered meeting minutes
                markdown_content = self._generate_bedrock_minutes(
                    intermediate_path, title, progress
                )

                # Step 6: Save output file
                self._save_output_file(output_file, markdown_content, progress)

            # Collect statistics if requested
            stats = None
            if self.config.stats:
                stats = self._collect_statistics(original_cues, cues, preprocessor)

            # Delete VTT file if requested and processing was successful
            if self.config.delete_vtt_file:
                self._delete_vtt_file(input_file)

            return ProcessingResult(
                success=True,
                input_file=input_file,
                output_file=output_file,
                intermediate_file=intermediate_path,
                statistics=stats,
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                input_file=input_file,
                error=str(e),
            )

    def _initialize_components(self) -> tuple[VTTParser, TextPreprocessor]:
        """Initialize parser and preprocessor components."""
        parser = VTTParser()
        config = PreprocessingConfig(
            min_duration=self.config.min_duration,
            merge_gap_threshold=self.config.merge_threshold,
            duplicate_threshold=self.config.duplicate_threshold,
            filler_words_file=self.config.filter_words_file,
            replacement_rules_file=self.config.replacement_rules_file,
        )
        preprocessor = TextPreprocessor(config)
        return parser, preprocessor

    def _parse_vtt_file(
        self, parser: VTTParser, input_file: Path, progress: Progress
    ) -> list[VTTCue]:
        """Parse VTT file and return cues."""
        task = progress.add_task("VTTファイルを解析中...", total=None)
        try:
            cues = parser.parse_file(input_file)
            progress.update(task, description="✓ VTTファイル解析完了")

            if self.config.verbose:
                self._display_parsing_results(parser, cues)

            return cues
        except Exception as e:
            progress.stop()
            raise RuntimeError(f"VTTファイルの解析に失敗しました: {e}") from e

    def _display_parsing_results(self, parser: VTTParser, cues: list[VTTCue]) -> None:
        """Display verbose parsing results."""
        self.console.print(f"解析されたキュー数: {len(cues)}")
        if cues:
            speakers = parser.get_speakers(cues)
            duration = parser.get_duration(cues)
            self.console.print(f"参加者数: {len(speakers)}")
            self.console.print(f"会議時間: {duration:.1f}秒")
            self.console.print(f"参加者: {', '.join(speakers)}")
        self.console.print()

    def _preprocess_cues(
        self, preprocessor: TextPreprocessor, cues: list[VTTCue], progress: Progress
    ) -> list[VTTCue]:
        """Preprocess cues if preprocessing is enabled."""
        if self.config.no_preprocessing or not cues:
            return cues

        original_count = len(cues)
        task = progress.add_task("テキストを前処理中...", total=None)

        try:
            processed_cues = preprocessor.preprocess_cues(cues)
            progress.update(task, description="✓ 前処理完了")

            if self.config.verbose:
                self.console.print(f"前処理後のキュー数: {len(processed_cues)}")
                removed = original_count - len(processed_cues)
                if removed > 0:
                    self.console.print(f"除去されたキュー数: {removed}")
                self.console.print()

            return processed_cues
        except Exception as e:
            progress.stop()
            raise RuntimeError(f"前処理に失敗しました: {e}") from e

    def _save_intermediate_file(
        self,
        cues: list[VTTCue],
        output: Path,
        intermediate_file: Path | None,
        title: str | None,
        progress: Progress,
    ) -> Path:
        """Save intermediate markdown file."""
        intermediate_path = intermediate_file or output.with_suffix(".preprocessed.md")

        def save_operation() -> Path:
            self._check_file_overwrite(intermediate_path, "Intermediate file")

            writer = IntermediateTranscriptWriter()
            stats = writer.get_statistics(cues)

            metadata: dict[str, Any] = {
                "participants": stats["speakers"],
                "duration": writer.format_duration(stats["duration"]),
            }

            writer.write_markdown(
                cues, intermediate_path, title or "前処理済み会議記録", metadata
            )

            if self.config.verbose:
                self.console.print(f"中間ファイル: {intermediate_path}")
                self.console.print(f"発言者数: {len(stats['speakers'])}名")
                self.console.print(f"総文字数: {stats['word_count']}文字")
                self.console.print()

            return intermediate_path

        return self._execute_with_progress(
            save_operation,
            "中間ファイルを保存中...",
            "✓ 中間ファイル保存完了",
            "中間ファイルの保存に失敗しました",
            progress,
        )

    def _generate_chat_prompt(
        self,
        intermediate_path: Path,
        chat_prompt_file: Path,
        title: str | None,
        progress: Progress,
    ) -> None:
        """Generate chat prompt file."""

        def generate_operation() -> None:
            self._check_file_overwrite(chat_prompt_file, "Chat prompt file")

            bedrock_generator = BedrockMeetingMinutesGenerator(
                region_name=self.config.bedrock_region,
                model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                prompt_template_file=self.config.prompt_template,
            )

            intermediate_content = intermediate_path.read_text(encoding="utf-8")
            meeting_title = title or "会議議事録"
            chat_prompt = bedrock_generator.create_chat_prompt(
                intermediate_content, meeting_title
            )

            chat_prompt_file.write_text(chat_prompt, encoding="utf-8")

            if self.config.verbose:
                self.console.print(f"チャットプロンプトファイル: {chat_prompt_file}")
                self.console.print()

        self._execute_with_progress(
            generate_operation,
            "チャットプロンプトファイルを生成中...",
            "✓ チャットプロンプトファイル生成完了",
            "チャットプロンプトファイルの生成に失敗しました",
            progress,
        )

    def _generate_bedrock_minutes(
        self, intermediate_path: Path, title: str | None, progress: Progress
    ) -> str:
        """Generate meeting minutes using Bedrock."""

        def generate_operation() -> str:
            self._validate_bedrock_config()

            bedrock_generator = self._create_bedrock_generator()

            intermediate_content = intermediate_path.read_text(encoding="utf-8")
            meeting_title = title or "会議議事録"
            markdown_content = bedrock_generator.generate_minutes_from_markdown(
                intermediate_content, title=meeting_title
            )

            if self.config.verbose:
                if self.config.bedrock_model:
                    self.console.print(f"Bedrockモデル: {self.config.bedrock_model}")
                if self.config.bedrock_inference_profile_id:
                    self.console.print(
                        f"Bedrock推論プロファイル: "
                        f"{self.config.bedrock_inference_profile_id}"
                    )
                self.console.print(f"リージョン: {self.config.bedrock_region}")
                self.console.print()

            return markdown_content

        try:
            return self._execute_with_progress(
                generate_operation,
                "AI議事録を生成中...",
                "✓ AI議事録生成完了",
                "議事録の生成に失敗しました",
                progress,
            )
        except RuntimeError as e:
            # Check if it's a BedrockError and re-raise with specific message
            if e.__cause__ and isinstance(e.__cause__, BedrockError):
                raise RuntimeError(f"Bedrock APIエラー: {e.__cause__}") from e.__cause__
            raise

    def _validate_bedrock_config(self) -> None:
        """Validate Bedrock configuration parameters."""
        if self.config.bedrock_model and self.config.bedrock_inference_profile_id:
            raise ValueError(
                "--bedrock-model と --bedrock-inference-profile-id の"
                "両方を指定することは"
                "できません。どちらか一方のみを指定してください。"
            )
        if (
            not self.config.bedrock_model
            and not self.config.bedrock_inference_profile_id
        ):
            raise ValueError(
                "--bedrock-model または --bedrock-inference-profile-id のどちらか一方を"
                "指定してください。"
            )

    def _save_output_file(self, output: Path, content: str, progress: Progress) -> None:
        """Save final output file."""
        task = progress.add_task("ファイルを保存中...", total=None)

        try:
            if output.exists() and not self.config.overwrite:
                raise FileExistsError(
                    f"Output file already exists: {output}. "
                    "Use --overwrite to overwrite existing files."
                )

            output.write_text(content, encoding="utf-8")
            progress.update(task, description="✓ 保存完了")
        except Exception as e:
            progress.stop()
            raise RuntimeError(f"ファイルの保存に失敗しました: {e}") from e

    def _collect_statistics(
        self,
        original_cues: list[VTTCue],
        processed_cues: list[VTTCue],
        preprocessor: TextPreprocessor,
    ) -> dict[str, Any]:
        """Collect processing statistics."""
        if self.config.no_preprocessing or not original_cues:
            return {}

        return preprocessor.get_statistics(original_cues, processed_cues)

    def _delete_vtt_file(self, vtt_file: Path) -> None:
        """Delete VTT file after successful processing.

        Args:
            vtt_file: Path to VTT file to delete

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            if self.config.verbose:
                self.console.print(f"[yellow]VTTファイルを削除しています: {vtt_file}[/yellow]")

            vtt_file.unlink()

            if self.config.verbose:
                self.console.print(f"[green]✓ VTTファイルを削除しました: {vtt_file}[/green]")
            else:
                self.console.print(f"[green]✓ VTTファイルを削除しました: {vtt_file.name}[/green]")

        except FileNotFoundError:
            if self.config.verbose:
                self.console.print(f"[yellow]VTTファイルが見つかりません: {vtt_file}[/yellow]")
        except PermissionError as e:
            error_msg = f"VTTファイルの削除に失敗しました（権限エラー）: {vtt_file}"
            if self.config.verbose:
                self.console.print(f"[red]{error_msg}[/red]")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"VTTファイルの削除に失敗しました: {e}"
            if self.config.verbose:
                self.console.print(f"[red]{error_msg}[/red]")
            raise RuntimeError(error_msg) from e

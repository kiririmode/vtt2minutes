"""Command-line interface for VTT2Minutes."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .batch import (
    BatchJob,
    BatchProcessor,
    display_vtt_files_table,
    interactive_job_configuration,
    review_and_confirm_jobs,
    scan_vtt_files,
)
from .bedrock import BedrockError, BedrockMeetingMinutesGenerator
from .intermediate import IntermediateTranscriptWriter
from .parser import VTTCue, VTTParser
from .preprocessor import PreprocessingConfig, TextPreprocessor
from .processor import ProcessingConfig

console = Console()

T = TypeVar("T")


class FileOperationError(Exception):
    """Exception raised for file operation errors."""

    pass


def _check_file_exists_and_overwrite(
    file_path: Path, overwrite: bool, file_type: str
) -> None:
    """Check if file exists and handle overwrite logic.

    Args:
        file_path: Path to check
        overwrite: Whether overwriting is allowed
        file_type: Description of the file type for error message

    Raises:
        FileOperationError: If file exists and overwrite is False
    """
    if file_path.exists() and not overwrite:
        raise FileOperationError(
            f"{file_type} already exists: {file_path}. "
            "Use --overwrite to overwrite existing files."
        )


def _display_header(verbose: bool, input_file: Path, output: Path) -> None:
    """Display application header and file information."""
    if verbose:
        console.print(
            Panel.fit(
                Text("VTT2Minutes - Teams Transcript Processor", style="bold blue"),
                style="blue",
            )
        )
        console.print(f"Input file: {input_file}")
        console.print(f"Output file: {output}")
        console.print()


class OperationConfig:
    """Configuration for operations with standard descriptions."""

    def __init__(self, task_desc: str, success_desc: str, error_prefix: str) -> None:
        self.task_desc = task_desc
        self.success_desc = success_desc
        self.error_prefix = error_prefix


def _execute_with_progress[T](
    progress: Progress,
    task_description: str,
    success_description: str,
    error_message: str,
    operation: Callable[[], T],
    verbose: bool = False,
    verbose_callback: Callable[[T], None] | None = None,
) -> T:
    """Execute an operation with progress tracking and error handling.

    Args:
        progress: Progress instance
        task_description: Description for the task while running
        success_description: Description when task completes
        error_message: Error message prefix if operation fails
        operation: Function to execute
        verbose: Whether to call verbose callback
        verbose_callback: Optional callback for verbose output (receives result)

    Returns:
        Result of the operation

    Raises:
        SystemExit: If operation fails
    """
    task = progress.add_task(task_description, total=None)
    try:
        result = operation()
        progress.update(task, description=success_description)

        if verbose and verbose_callback:
            verbose_callback(result)

        return result
    except FileOperationError as e:
        progress.stop()
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        progress.stop()
        console.print(f"[red]{error_message}: {e}[/red]")
        sys.exit(1)


def _execute_configured_operation[T](
    config: OperationConfig,
    progress: Progress,
    operation: Callable[[], T],
    verbose: bool = False,
    verbose_callback: Callable[[T], None] | None = None,
) -> T:
    """Execute operation using a configuration object."""
    return _execute_with_progress(
        progress=progress,
        task_description=config.task_desc,
        success_description=config.success_desc,
        error_message=config.error_prefix,
        operation=operation,
        verbose=verbose,
        verbose_callback=verbose_callback,
    )


_PARSING_CONFIG = OperationConfig(
    task_desc="VTTファイルを解析中...",
    success_desc="✓ VTTファイル解析完了",
    error_prefix="VTTファイルの解析に失敗しました",
)

_SAVE_CONFIG = OperationConfig(
    task_desc="中間ファイルを保存中...",
    success_desc="✓ 中間ファイル保存完了",
    error_prefix="中間ファイルの保存に失敗しました",
)


def _initialize_components(
    min_duration: float,
    merge_threshold: float,
    duplicate_threshold: float,
    filter_words_file: Path | None,
    replacement_rules_file: Path | None,
) -> tuple[VTTParser, TextPreprocessor]:
    """Initialize parser and preprocessor components."""
    parser = VTTParser()
    config = PreprocessingConfig(
        min_duration=min_duration,
        merge_gap_threshold=merge_threshold,
        duplicate_threshold=duplicate_threshold,
        filler_words_file=filter_words_file,
        replacement_rules_file=replacement_rules_file,
    )
    preprocessor = TextPreprocessor(config)
    return parser, preprocessor


def _parse_vtt_file(
    parser: VTTParser, input_file: Path, progress: Progress, verbose: bool
) -> list[VTTCue]:
    """Parse VTT file and return cues."""
    return _execute_configured_operation(
        config=_PARSING_CONFIG,
        progress=progress,
        operation=lambda: parser.parse_file(input_file),
        verbose=verbose,
        verbose_callback=lambda result: _display_parsing_results(parser, result),
    )


def _display_parsing_results(parser: VTTParser, cues: list[VTTCue]) -> None:
    """Display verbose parsing results.

    Args:
        parser: VTT parser instance
        cues: Parsed VTT cues
    """
    console.print(f"解析されたキュー数: {len(cues)}")
    if cues:
        speakers = parser.get_speakers(cues)
        duration = parser.get_duration(cues)
        console.print(f"参加者数: {len(speakers)}")
        console.print(f"会議時間: {duration:.1f}秒")
        console.print(f"参加者: {', '.join(speakers)}")
    console.print()


def _preprocess_cues(
    preprocessor: TextPreprocessor,
    cues: list[VTTCue],
    no_preprocessing: bool,
    progress: Progress,
    verbose: bool,
) -> list[VTTCue]:
    """Preprocess cues if preprocessing is enabled."""
    if no_preprocessing or not cues:
        return cues

    original_count = len(cues)

    def verbose_callback(processed_cues: list[VTTCue]) -> None:
        console.print(f"前処理後のキュー数: {len(processed_cues)}")
        removed = original_count - len(processed_cues)
        if removed > 0:
            console.print(f"除去されたキュー数: {removed}")
        console.print()

    return _execute_with_progress(
        progress,
        "テキストを前処理中...",
        "✓ 前処理完了",
        "前処理に失敗しました",
        lambda: preprocessor.preprocess_cues(cues),
        verbose=verbose,
        verbose_callback=verbose_callback,
    )


def _save_intermediate_file(
    cues: list[VTTCue],
    output: Path,
    intermediate_file: Path | None,
    title: str | None,
    progress: Progress,
    verbose: bool,
    overwrite: bool = False,
) -> Path:
    """Save intermediate markdown file."""
    return _execute_configured_operation(
        config=_SAVE_CONFIG,
        progress=progress,
        operation=lambda: _save_intermediate_operation(
            cues, output, intermediate_file, title, overwrite
        ),
        verbose=verbose,
        verbose_callback=lambda result: _display_intermediate_stats(cues, result),
    )


def _save_intermediate_operation(
    cues: list[VTTCue],
    output: Path,
    intermediate_file: Path | None,
    title: str | None,
    overwrite: bool = False,
) -> Path:
    """Execute intermediate file save operation.

    Args:
        cues: VTT cues to save
        output: Output path
        intermediate_file: Optional intermediate file path
        title: Optional title
        overwrite: Whether to overwrite existing files

    Returns:
        Path to saved intermediate file

    Raises:
        FileOperationError: If intermediate file exists and overwrite is False
    """
    intermediate_path = intermediate_file or output.with_suffix(".preprocessed.md")
    _check_file_exists_and_overwrite(intermediate_path, overwrite, "Intermediate file")

    writer = IntermediateTranscriptWriter()
    stats = writer.get_statistics(cues)

    metadata: dict[str, Any] = {
        "participants": stats["speakers"],
        "duration": writer.format_duration(stats["duration"]),
    }

    writer.write_markdown(
        cues, intermediate_path, title or "前処理済み会議記録", metadata
    )
    return intermediate_path


def _display_intermediate_stats(cues: list[VTTCue], intermediate_path: Path) -> None:
    """Display intermediate file statistics.

    Args:
        cues: VTT cues
        intermediate_path: Path to intermediate file
    """
    writer = IntermediateTranscriptWriter()
    stats = writer.get_statistics(cues)
    console.print(f"中間ファイル: {intermediate_path}")
    console.print(f"発言者数: {len(stats['speakers'])}名")
    console.print(f"総文字数: {stats['word_count']}文字")
    console.print()


def _generate_chat_prompt(
    intermediate_path: Path,
    chat_prompt_file: Path,
    title: str | None,
    bedrock_region: str,
    prompt_template: Path | None,
    progress: Progress,
    verbose: bool,
    overwrite: bool = False,
) -> None:
    """Generate chat prompt file and exit."""

    def generate_operation() -> None:
        _check_file_exists_and_overwrite(
            chat_prompt_file, overwrite, "Chat prompt file"
        )

        bedrock_generator = BedrockMeetingMinutesGenerator(
            region_name=bedrock_region,
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            prompt_template_file=prompt_template,
        )

        intermediate_content = intermediate_path.read_text(encoding="utf-8")
        meeting_title = title or "会議議事録"
        chat_prompt = bedrock_generator.create_chat_prompt(
            intermediate_content, meeting_title
        )

        chat_prompt_file.write_text(chat_prompt, encoding="utf-8")

    def verbose_callback(_: None) -> None:
        console.print(f"チャットプロンプトファイル: {chat_prompt_file}")
        console.print()

    _execute_with_progress(
        progress,
        "チャットプロンプトファイルを生成中...",
        "✓ チャットプロンプトファイル生成完了",
        "チャットプロンプトファイルの生成に失敗しました",
        generate_operation,
        verbose=verbose,
        verbose_callback=verbose_callback,
    )

    _display_chat_prompt_success(chat_prompt_file, progress)


def _display_chat_prompt_success(chat_prompt_file: Path, progress: Progress) -> None:
    """Display success message for chat prompt generation."""
    progress.stop()
    console.print(
        Panel.fit(
            Text(
                "チャットプロンプトファイルが正常に生成されました",
                style="bold green",
            ),
            style="green",
        )
    )
    console.print(f"[green]チャットプロンプトファイル: {chat_prompt_file}[/green]")
    console.print(
        "[yellow]このファイルをChatGPTなどのチャット型AIサービスに"
        "コピー&ペーストしてください。[/yellow]"
    )


def _validate_bedrock_config(
    bedrock_model: str | None, bedrock_inference_profile_id: str | None
) -> None:
    """Validate Bedrock configuration parameters."""
    if bedrock_model and bedrock_inference_profile_id:
        console.print(
            "[red]エラー: --bedrock-model と "
            "--bedrock-inference-profile-id の両方を指定することは"
            "できません。どちらか一方のみを指定してください。[/red]"
        )
        sys.exit(1)
    if not bedrock_model and not bedrock_inference_profile_id:
        console.print(
            "[red]エラー: --bedrock-model または "
            "--bedrock-inference-profile-id のどちらか一方を"
            "指定してください。[/red]"
        )
        sys.exit(1)


def _generate_bedrock_minutes(
    intermediate_path: Path,
    title: str | None,
    bedrock_model: str | None,
    bedrock_inference_profile_id: str | None,
    bedrock_region: str,
    prompt_template: Path | None,
    progress: Progress,
    verbose: bool,
) -> str:
    """Generate meeting minutes using Bedrock."""
    task = progress.add_task("AI議事録を生成中...", total=None)
    try:
        _validate_bedrock_config(bedrock_model, bedrock_inference_profile_id)

        bedrock_generator = BedrockMeetingMinutesGenerator(
            region_name=bedrock_region,
            model_id=bedrock_model,
            inference_profile_id=bedrock_inference_profile_id,
            prompt_template_file=prompt_template,
        )

        intermediate_content = intermediate_path.read_text(encoding="utf-8")
        meeting_title = title or "会議議事録"
        markdown_content = bedrock_generator.generate_minutes_from_markdown(
            intermediate_content, title=meeting_title
        )
        progress.update(task, description="✓ AI議事録生成完了")

        if verbose:
            if bedrock_model:
                console.print(f"Bedrockモデル: {bedrock_model}")
            if bedrock_inference_profile_id:
                console.print(
                    f"Bedrock推論プロファイル: {bedrock_inference_profile_id}"
                )
            console.print(f"リージョン: {bedrock_region}")
            console.print()

        return markdown_content
    except BedrockError as e:
        progress.stop()
        console.print(f"[red]Bedrock APIエラー: {e}[/red]")
        console.print("[yellow]AWS認証情報とBedrock権限を確認してください。[/yellow]")
        sys.exit(1)
    except Exception as e:
        progress.stop()
        console.print(f"[red]議事録の生成に失敗しました: {e}[/red]")
        sys.exit(1)


_OUTPUT_SAVE_CONFIG = OperationConfig(
    task_desc="ファイルを保存中...",
    success_desc="✓ 保存完了",
    error_prefix="ファイルの保存に失敗しました",
)


def _save_output_file(
    output: Path, content: str, progress: Progress, overwrite: bool = False
) -> None:
    """Save final output file.

    Args:
        output: Output file path
        content: Content to write
        progress: Progress instance
        overwrite: Whether to overwrite existing files

    Raises:
        FileExistsError: If output file exists and overwrite is False
    """
    _execute_configured_operation(
        config=_OUTPUT_SAVE_CONFIG,
        progress=progress,
        operation=lambda: _save_file_operation(output, content, overwrite),
        verbose=False,
    )


def _save_file_operation(file_path: Path, content: str, overwrite: bool) -> None:
    """Execute file save operation with overwrite checking."""
    _check_file_exists_and_overwrite(file_path, overwrite, "Output file")
    file_path.write_text(content, encoding="utf-8")


def _display_statistics(
    stats: bool,
    no_preprocessing: bool,
    original_cues: list[VTTCue],
    processed_cues: list[VTTCue],
    preprocessor: TextPreprocessor,
) -> None:
    """Display preprocessing statistics if requested."""
    if not (stats and not no_preprocessing and original_cues):
        return

    preprocessing_stats = preprocessor.get_statistics(original_cues, processed_cues)

    console.print("\n" + "=" * 50)
    console.print("[bold]前処理統計[/bold]")
    console.print("=" * 50)
    console.print(f"元のキュー数: {preprocessing_stats['original_count']}")
    console.print(f"処理後キュー数: {preprocessing_stats['processed_count']}")
    console.print(f"削除キュー数: {preprocessing_stats['removed_count']}")
    console.print(f"削除率: {preprocessing_stats['removal_rate']:.1%}")
    console.print(f"元の総文字数: {preprocessing_stats['original_text_length']:,}")
    console.print(f"処理後総文字数: {preprocessing_stats['processed_text_length']:,}")

    duration_reduction = (
        preprocessing_stats["original_duration"]
        - preprocessing_stats["processed_duration"]
    )
    console.print(f"短縮時間: {duration_reduction:.1f}秒")


def _display_final_summary(output: Path, title: str | None, cues: list[VTTCue]) -> None:
    """Display final success message and summary."""
    console.print()
    console.print(
        Panel.fit(
            f"[green]✓ 議事録が正常に作成されました: {output}[/green]",
            style="green",
        )
    )

    meeting_title = title or "会議議事録"
    console.print(f"\n[bold]{meeting_title}[/bold]")
    console.print("AI議事録生成が完了しました")
    if cues:
        writer = IntermediateTranscriptWriter()
        summary_stats = writer.get_statistics(cues)
        console.print(f"参加者: {len(summary_stats['speakers'])}名")
        console.print(f"会議時間: {writer.format_duration(summary_stats['duration'])}")
        console.print(f"総文字数: {summary_stats['word_count']}文字")


def _delete_vtt_file(vtt_file: Path, verbose: bool) -> None:
    """Delete VTT file after successful processing.

    Args:
        vtt_file: Path to VTT file to delete
        verbose: Whether to display deletion message

    Note:
        If deletion fails, a warning is logged but no exception is raised
    """
    try:
        vtt_file.unlink()
        if verbose:
            console.print(f"✓ VTTファイルを削除しました: {vtt_file}")
    except Exception as e:
        # Log warning but don't fail the entire operation
        console.print(
            f"[yellow]警告: VTTファイルの削除に失敗しました: {vtt_file}[/yellow]"
        )
        console.print(f"[yellow]エラー: {e}[/yellow]")


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: input_file.md)",
)
@click.option("--title", "-t", type=str, help="Meeting title for the minutes")
@click.option("--no-preprocessing", is_flag=True, help="Skip text preprocessing step")
@click.option(
    "--min-duration",
    type=float,
    default=0.5,
    help="Minimum cue duration in seconds (default: 0.5)",
)
@click.option(
    "--merge-threshold",
    type=float,
    default=2.0,
    help="Time gap threshold for merging cues in seconds (default: 2.0)",
)
@click.option(
    "--duplicate-threshold",
    type=float,
    default=0.8,
    help="Similarity threshold for duplicate detection (0.0-1.0, default: 0.8)",
)
@click.option(
    "--filter-words-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom filler words file",
)
@click.option(
    "--replacement-rules-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom word replacement rules file",
)
@click.option(
    "--bedrock-model",
    type=str,
    help=(
        "Bedrock model ID to use "
        "(mutually exclusive with --bedrock-inference-profile-id)"
    ),
)
@click.option(
    "--bedrock-inference-profile-id",
    type=str,
    help=(
        "Bedrock inference profile ID to use (mutually exclusive with --bedrock-model)"
    ),
)
@click.option(
    "--bedrock-region",
    type=str,
    default="ap-northeast-1",
    help="AWS region for Bedrock (default: ap-northeast-1)",
)
@click.option(
    "--intermediate-file",
    type=click.Path(path_type=Path),
    help="Path to save intermediate preprocessed file",
)
@click.option(
    "--prompt-template",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom prompt template file",
)
@click.option(
    "--chat-prompt-file",
    type=click.Path(path_type=Path),
    help=(
        "Output chat prompt to file for use with ChatGPT-like services (skips Bedrock)"
    ),
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--stats", is_flag=True, help="Show preprocessing statistics")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files without warning",
)
@click.option(
    "--keep-vtt",
    is_flag=True,
    help="Keep VTT files after successful processing",
)
def main(
    input_file: Path,
    output: Path | None,
    title: str | None,
    no_preprocessing: bool,
    min_duration: float,
    merge_threshold: float,
    duplicate_threshold: float,
    filter_words_file: Path | None,
    replacement_rules_file: Path | None,
    bedrock_model: str | None,
    bedrock_inference_profile_id: str | None,
    bedrock_region: str,
    intermediate_file: Path | None,
    prompt_template: Path | None,
    chat_prompt_file: Path | None,
    verbose: bool,
    stats: bool,
    overwrite: bool,
    keep_vtt: bool,
) -> None:
    """Convert Microsoft Teams VTT transcripts to AI-powered meeting minutes.

    This tool processes VTT (WebVTT) transcript files from Microsoft Teams
    and generates AI-powered meeting minutes using Amazon Bedrock, or outputs
    a chat prompt file for use with external AI services like ChatGPT.

    For Bedrock usage, you must specify either --bedrock-model or
    --bedrock-inference-profile-id, but not both.

    Example usage:

        vtt2minutes meeting.vtt

        vtt2minutes meeting.vtt -o minutes.md -t "Project Planning Meeting"

        vtt2minutes meeting.vtt --bedrock-model \\
            anthropic.claude-3-5-sonnet-20241022-v2:0

        vtt2minutes meeting.vtt --chat-prompt-file prompt.txt
    """
    try:
        # Determine output file path
        if output is None:
            output = input_file.with_suffix(".md")

        # Display header and initialize components
        _display_header(verbose, input_file, output)
        parser, preprocessor = _initialize_components(
            min_duration,
            merge_threshold,
            duplicate_threshold,
            filter_words_file,
            replacement_rules_file,
        )

        # Process with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Step 1: Parse VTT file
            cues = _parse_vtt_file(parser, input_file, progress, verbose)

            # Step 2: Preprocess (if enabled)
            original_cues = cues.copy()
            cues = _preprocess_cues(
                preprocessor, cues, no_preprocessing, progress, verbose
            )

            # Step 3: Save intermediate file
            intermediate_path = _save_intermediate_file(
                cues, output, intermediate_file, title, progress, verbose, overwrite
            )

            # Step 4: Generate chat prompt file if requested (skips Bedrock)
            if chat_prompt_file:
                _generate_chat_prompt(
                    intermediate_path,
                    chat_prompt_file,
                    title,
                    bedrock_region,
                    prompt_template,
                    progress,
                    verbose,
                    overwrite,
                )
                # Delete VTT file after successful processing if requested
                if not keep_vtt:
                    _delete_vtt_file(input_file, verbose)
                return

            # Step 5: Generate AI-powered meeting minutes using Bedrock
            markdown_content = _generate_bedrock_minutes(
                intermediate_path,
                title,
                bedrock_model,
                bedrock_inference_profile_id,
                bedrock_region,
                prompt_template,
                progress,
                verbose,
            )

            # Step 6: Save output file
            _save_output_file(output, markdown_content, progress, overwrite)

        # Delete VTT file after successful processing if requested
        if not keep_vtt:
            _delete_vtt_file(input_file, verbose)

        # Show statistics if requested
        _display_statistics(stats, no_preprocessing, original_cues, cues, preprocessor)

        # Display final summary
        _display_final_summary(output, title, cues)

    except KeyboardInterrupt:
        console.print("\n[yellow]処理が中断されました。[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]予期しないエラーが発生しました: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@click.group()
@click.version_option(version="1.0.0", prog_name="vtt2minutes")
def cli() -> None:
    """VTT2Minutes - Convert Teams VTT transcripts to meeting minutes."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
def info(input_file: Path) -> None:
    """Show information about a VTT file without processing it."""
    try:
        parser = VTTParser()
        cues = parser.parse_file(input_file)

        _display_info_header(input_file)

        if not cues:
            console.print("[yellow]ファイルにキューが見つかりませんでした。[/yellow]")
            return

        _display_basic_info(parser, cues)
        _display_speaker_info(parser, cues)
        _display_sample_cues(cues)

    except Exception as e:
        console.print(f"[red]ファイル情報の取得に失敗しました: {e}[/red]")
        sys.exit(1)


def _display_info_header(input_file: Path) -> None:
    """Display the information header panel.

    Args:
        input_file: Path to the VTT file
    """
    console.print(
        Panel.fit(
            Text(f"VTT File Information: {input_file.name}", style="bold blue"),
            style="blue",
        )
    )


def _display_basic_info(parser: VTTParser, cues: list[VTTCue]) -> None:
    """Display basic file information.

    Args:
        parser: VTT parser instance
        cues: List of parsed VTT cues
    """
    speakers = parser.get_speakers(cues)
    duration = parser.get_duration(cues)
    total_text = sum(len(cue.text) for cue in cues)

    console.print("[bold]基本情報[/bold]")
    console.print(f"  キュー数: {len(cues):,}")
    console.print(f"  参加者数: {len(speakers)}")
    console.print(f"  総時間: {duration:.1f}秒 ({duration / 60:.1f}分)")
    console.print(f"  総文字数: {total_text:,}")
    console.print(f"  平均キュー長: {total_text / len(cues):.1f}文字")


def _display_speaker_info(parser: VTTParser, cues: list[VTTCue]) -> None:
    """Display information about speakers.

    Args:
        parser: VTT parser instance
        cues: List of parsed VTT cues
    """
    speakers = parser.get_speakers(cues)
    if not speakers:
        return

    console.print("\n[bold]参加者[/bold]")
    for speaker in speakers:
        speaker_cues = parser.filter_by_speaker(cues, speaker)
        speaker_text = sum(len(cue.text) for cue in speaker_cues)
        speaker_duration = sum(cue.duration for cue in speaker_cues)
        console.print(
            f"  {speaker}: {len(speaker_cues)}発言, "
            f"{speaker_text:,}文字, {speaker_duration:.1f}秒"
        )


def _display_sample_cues(cues: list[VTTCue]) -> None:
    """Display sample cues from the beginning of the file.

    Args:
        cues: List of parsed VTT cues
    """
    console.print("\n[bold]サンプル（最初の3キュー）[/bold]")
    for i, cue in enumerate(cues[:3], 1):
        speaker_part = f"{cue.speaker}: " if cue.speaker else ""
        text_preview = cue.text[:100] + "..." if len(cue.text) > 100 else cue.text
        console.print(f"  {i}. [{cue.start_time}] {speaker_part}{text_preview}")


# Add the info command to the main CLI
cli.add_command(info)


def _setup_batch_processing(
    output_dir: Path | None, directory: Path, recursive: bool
) -> list[Path]:
    """Set up batch processing and scan for VTT files."""
    # Create output directory if specified
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Display header and scan for VTT files
    console.print()
    console.print(
        Panel.fit(Text("VTT2Minutes バッチ処理", style="bold blue"), style="blue")
    )
    console.print(f"ディレクトリをスキャン中: {directory}")

    vtt_files = scan_vtt_files(directory, recursive=recursive)
    display_vtt_files_table(vtt_files, directory)

    return vtt_files


def _configure_batch_jobs(
    vtt_files: list[Path], directory: Path, output_dir: Path | None
) -> list[BatchJob]:
    """Configure batch jobs interactively."""
    if not vtt_files:
        return []

    jobs = interactive_job_configuration(vtt_files, directory, output_dir)

    if not jobs:
        console.print("[yellow]処理対象のファイルがありません。[/yellow]")
        return []

    # Review and confirm jobs
    proceed, final_jobs = review_and_confirm_jobs(jobs, directory)

    if not proceed or not final_jobs:
        console.print("[yellow]処理をキャンセルしました。[/yellow]")
        return []

    return final_jobs


def _create_processing_config(
    no_preprocessing: bool,
    min_duration: float,
    merge_threshold: float,
    duplicate_threshold: float,
    filter_words_file: Path | None,
    replacement_rules_file: Path | None,
    bedrock_model: str | None,
    bedrock_inference_profile_id: str | None,
    bedrock_region: str,
    prompt_template: Path | None,
    overwrite: bool,
    verbose: bool,
    stats: bool,
    keep_vtt: bool,
) -> ProcessingConfig:
    """Create processing configuration."""
    return ProcessingConfig(
        no_preprocessing=no_preprocessing,
        min_duration=min_duration,
        merge_threshold=merge_threshold,
        duplicate_threshold=duplicate_threshold,
        filter_words_file=filter_words_file,
        replacement_rules_file=replacement_rules_file,
        bedrock_model=bedrock_model,
        bedrock_inference_profile_id=bedrock_inference_profile_id,
        bedrock_region=bedrock_region,
        prompt_template=prompt_template,
        overwrite=overwrite,
        verbose=verbose,
        stats=stats,
        keep_vtt=keep_vtt,
    )


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for generated files",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Scan directory recursively (default: --recursive)",
)
@click.option("--title", "-t", type=str, help="Default meeting title prefix")
@click.option("--no-preprocessing", is_flag=True, help="Skip text preprocessing step")
@click.option(
    "--min-duration",
    type=float,
    default=0.5,
    help="Minimum cue duration in seconds (default: 0.5)",
)
@click.option(
    "--merge-threshold",
    type=float,
    default=2.0,
    help="Time gap threshold for merging cues in seconds (default: 2.0)",
)
@click.option(
    "--duplicate-threshold",
    type=float,
    default=0.8,
    help="Similarity threshold for duplicate detection (0.0-1.0, default: 0.8)",
)
@click.option(
    "--filter-words-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom filler words file",
)
@click.option(
    "--replacement-rules-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom word replacement rules file",
)
@click.option(
    "--bedrock-model",
    type=str,
    help=(
        "Bedrock model ID to use "
        "(mutually exclusive with --bedrock-inference-profile-id)"
    ),
)
@click.option(
    "--bedrock-inference-profile-id",
    type=str,
    help=(
        "Bedrock inference profile ID to use (mutually exclusive with --bedrock-model)"
    ),
)
@click.option(
    "--bedrock-region",
    type=str,
    default="ap-northeast-1",
    help="AWS region for Bedrock (default: ap-northeast-1)",
)
@click.option(
    "--prompt-template",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom prompt template file",
)
@click.option(
    "--chat-prompt-files",
    is_flag=True,
    help="Generate chat prompt files instead of using Bedrock",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--stats", is_flag=True, help="Show preprocessing statistics")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files without warning",
)
@click.option(
    "--keep-vtt",
    is_flag=True,
    help="Keep VTT files after successful processing",
)
def batch(
    directory: Path,
    output_dir: Path | None,
    recursive: bool,
    title: str | None,
    no_preprocessing: bool,
    min_duration: float,
    merge_threshold: float,
    duplicate_threshold: float,
    filter_words_file: Path | None,
    replacement_rules_file: Path | None,
    bedrock_model: str | None,
    bedrock_inference_profile_id: str | None,
    bedrock_region: str,
    prompt_template: Path | None,
    chat_prompt_files: bool,
    verbose: bool,
    stats: bool,
    overwrite: bool,
    keep_vtt: bool,
) -> None:
    """Process multiple VTT files in a directory interactively.

    This command scans the specified directory for VTT files and allows you to
    configure and process them in batch mode. Each file can be individually
    configured with its own title and output path.

    Example usage:

        vtt2minutes batch ./meetings

        vtt2minutes batch ./meetings --output-dir ./output

        vtt2minutes batch ./meetings --bedrock-model \
            anthropic.claude-3-5-sonnet-20241022-v2:0
    """
    try:
        # Validate Bedrock configuration if not using chat prompt files
        if not chat_prompt_files:
            _validate_bedrock_config(bedrock_model, bedrock_inference_profile_id)

        # Setup and scan for files
        vtt_files = _setup_batch_processing(output_dir, directory, recursive)

        # Configure batch jobs
        final_jobs = _configure_batch_jobs(vtt_files, directory, output_dir)
        if not final_jobs:
            return

        # Create configuration and process
        config = _create_processing_config(
            no_preprocessing,
            min_duration,
            merge_threshold,
            duplicate_threshold,
            filter_words_file,
            replacement_rules_file,
            bedrock_model,
            bedrock_inference_profile_id,
            bedrock_region,
            prompt_template,
            overwrite,
            verbose,
            stats,
            keep_vtt,
        )

        processor = BatchProcessor(config)

        if chat_prompt_files:
            _process_batch_for_chat_prompts(processor, final_jobs)
        else:
            processor.process_batch(final_jobs)

    except KeyboardInterrupt:
        console.print("\n[yellow]処理が中断されました。[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]予期しないエラーが発生しました: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _process_batch_for_chat_prompts(
    processor: BatchProcessor, jobs: list[BatchJob]
) -> None:
    """Process batch jobs to generate chat prompt files."""

    # Modify jobs to generate chat prompt files
    chat_jobs: list[BatchJob] = []
    for job in jobs:
        # Create chat prompt filename
        chat_prompt_file = job.output_file.with_suffix(".txt")

        # Create a modified job for chat prompt generation
        modified_job = BatchJob(
            vtt_file=job.vtt_file,
            title=job.title,
            output_file=chat_prompt_file,  # This will be detected in BatchProcessor
            enabled=job.enabled,
            intermediate_file=job.intermediate_file,
        )
        chat_jobs.append(modified_job)

    # Process with the modified jobs
    processor.process_batch(chat_jobs)


if __name__ == "__main__":
    main()

"""Command-line interface for VTT2Minutes."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .parser import VTTParser
from .preprocessor import PreprocessingConfig, TextPreprocessor
from .summarizer import MeetingSummarizer

console = Console()


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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--stats", is_flag=True, help="Show preprocessing statistics")
def main(
    input_file: Path,
    output: Path | None,
    title: str | None,
    no_preprocessing: bool,
    min_duration: float,
    merge_threshold: float,
    duplicate_threshold: float,
    filter_words_file: Path | None,
    verbose: bool,
    stats: bool,
) -> None:
    """Convert Microsoft Teams VTT transcripts to structured meeting minutes.

    This tool processes VTT (WebVTT) transcript files from Microsoft Teams
    and generates well-structured meeting minutes in Markdown format.

    Example usage:

        vtt2minutes meeting.vtt

        vtt2minutes meeting.vtt -o minutes.md -t "Project Planning Meeting"

        vtt2minutes meeting.vtt --no-preprocessing --verbose
    """
    try:
        # Determine output file path
        if output is None:
            output = input_file.with_suffix(".md")

        # Display header
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

        # Initialize components
        parser = VTTParser()

        # Initialize preprocessor (even if not used, to avoid unbound variable)
        config = PreprocessingConfig(
            min_duration=min_duration,
            merge_gap_threshold=merge_threshold,
            duplicate_threshold=duplicate_threshold,
            filler_words_file=filter_words_file,
        )
        preprocessor = TextPreprocessor(config)

        summarizer = MeetingSummarizer()

        # Process with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Step 1: Parse VTT file
            task1 = progress.add_task("VTTファイルを解析中...", total=None)
            try:
                cues = parser.parse_file(input_file)
                progress.update(task1, description="✓ VTTファイル解析完了")

                if verbose:
                    console.print(f"解析されたキュー数: {len(cues)}")
                    if cues:
                        speakers = parser.get_speakers(cues)
                        duration = parser.get_duration(cues)
                        console.print(f"参加者数: {len(speakers)}")
                        console.print(f"会議時間: {duration:.1f}秒")
                        console.print(f"参加者: {', '.join(speakers)}")
                    console.print()

            except Exception as e:
                progress.stop()
                console.print(f"[red]VTTファイルの解析に失敗しました: {e}[/red]")
                sys.exit(1)

            # Step 2: Preprocess (if enabled)
            original_cues = cues.copy()
            if not no_preprocessing and cues:
                task2 = progress.add_task("テキストを前処理中...", total=None)
                try:
                    cues = preprocessor.preprocess_cues(cues)
                    progress.update(task2, description="✓ 前処理完了")

                    if verbose:
                        console.print(f"前処理後のキュー数: {len(cues)}")
                        removed = len(original_cues) - len(cues)
                        if removed > 0:
                            console.print(f"除去されたキュー数: {removed}")
                        console.print()

                except Exception as e:
                    progress.stop()
                    console.print(f"[red]前処理に失敗しました: {e}[/red]")
                    sys.exit(1)

            # Step 3: Generate summary
            task3 = progress.add_task("議事録を生成中...", total=None)
            try:
                summary = summarizer.create_summary(cues, title)
                progress.update(task3, description="✓ 議事録生成完了")

                if verbose:
                    console.print(f"決定事項: {len(summary.decisions)}件")
                    console.print(f"アクションアイテム: {len(summary.action_items)}件")
                    console.print(f"主要ポイント: {len(summary.key_points)}件")
                    console.print()

            except Exception as e:
                progress.stop()
                console.print(f"[red]議事録の生成に失敗しました: {e}[/red]")
                sys.exit(1)

            # Step 4: Write output
            task4 = progress.add_task("ファイルを保存中...", total=None)
            try:
                markdown_content = summary.to_markdown()
                output.write_text(markdown_content, encoding="utf-8")
                progress.update(task4, description="✓ 保存完了")

            except Exception as e:
                progress.stop()
                console.print(f"[red]ファイルの保存に失敗しました: {e}[/red]")
                sys.exit(1)

        # Show statistics if requested
        if stats and not no_preprocessing and original_cues:
            preprocessing_stats = preprocessor.get_statistics(original_cues, cues)

            console.print("\n" + "=" * 50)
            console.print("[bold]前処理統計[/bold]")
            console.print("=" * 50)
            console.print(f"元のキュー数: {preprocessing_stats['original_count']}")
            console.print(f"処理後キュー数: {preprocessing_stats['processed_count']}")
            console.print(f"削除キュー数: {preprocessing_stats['removed_count']}")
            console.print(f"削除率: {preprocessing_stats['removal_rate']:.1%}")
            console.print(
                f"元の総文字数: {preprocessing_stats['original_text_length']:,}"
            )
            console.print(
                f"処理後総文字数: {preprocessing_stats['processed_text_length']:,}"
            )

            duration_reduction = (
                preprocessing_stats["original_duration"]
                - preprocessing_stats["processed_duration"]
            )
            console.print(f"短縮時間: {duration_reduction:.1f}秒")

        # Success message
        console.print()
        console.print(
            Panel.fit(
                f"[green]✓ 議事録が正常に作成されました: {output}[/green]",
                style="green",
            )
        )

        # Show brief summary
        console.print(f"\n[bold]{summary.title}[/bold]")
        console.print(f"参加者: {len(summary.participants)}名")
        console.print(f"会議時間: {summary.duration}")
        console.print(f"決定事項: {len(summary.decisions)}件")
        console.print(f"アクションアイテム: {len(summary.action_items)}件")

    except KeyboardInterrupt:
        console.print("\n[yellow]処理が中断されました。[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]予期しないエラーが発生しました: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@click.group()
@click.version_option(version="0.1.0", prog_name="vtt2minutes")
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

        console.print(
            Panel.fit(
                Text(f"VTT File Information: {input_file.name}", style="bold blue"),
                style="blue",
            )
        )

        if not cues:
            console.print("[yellow]ファイルにキューが見つかりませんでした。[/yellow]")
            return

        speakers = parser.get_speakers(cues)
        duration = parser.get_duration(cues)
        total_text = sum(len(cue.text) for cue in cues)

        console.print("[bold]基本情報[/bold]")
        console.print(f"  キュー数: {len(cues):,}")
        console.print(f"  参加者数: {len(speakers)}")
        console.print(f"  総時間: {duration:.1f}秒 ({duration / 60:.1f}分)")
        console.print(f"  総文字数: {total_text:,}")
        console.print(f"  平均キュー長: {total_text / len(cues):.1f}文字")

        if speakers:
            console.print("\n[bold]参加者[/bold]")
            for speaker in speakers:
                speaker_cues = parser.filter_by_speaker(cues, speaker)
                speaker_text = sum(len(cue.text) for cue in speaker_cues)
                speaker_duration = sum(cue.duration for cue in speaker_cues)
                console.print(
                    f"  {speaker}: {len(speaker_cues)}発言, "
                    f"{speaker_text:,}文字, {speaker_duration:.1f}秒"
                )

        # Show first few cues as examples
        console.print("\n[bold]サンプル（最初の3キュー）[/bold]")
        for i, cue in enumerate(cues[:3], 1):
            speaker_part = f"{cue.speaker}: " if cue.speaker else ""
            text_preview = cue.text[:100] + "..." if len(cue.text) > 100 else cue.text
            console.print(f"  {i}. [{cue.start_time}] {speaker_part}{text_preview}")

    except Exception as e:
        console.print(f"[red]ファイル情報の取得に失敗しました: {e}[/red]")
        sys.exit(1)


# Add the info command to the main CLI
cli.add_command(info)

if __name__ == "__main__":
    main()

"""Batch processing functionality for VTT2Minutes."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .processor import ProcessingConfig, ProcessingResult, VTTFileProcessor


def scan_vtt_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Scan directory for VTT files.

    Args:
        directory: Directory to scan
        recursive: Whether to scan recursively

    Returns:
        List of VTT file paths sorted by name

    Raises:
        FileNotFoundError: If directory does not exist
        NotADirectoryError: If path is not a directory
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    pattern = "**/*.vtt" if recursive else "*.vtt"
    vtt_files: list[Path] = []

    for file_path in directory.glob(pattern):
        # Skip hidden files and directories
        if any(part.startswith(".") for part in file_path.parts):
            continue

        # Only include actual files (not directories)
        if file_path.is_file():
            vtt_files.append(file_path)

    return sorted(vtt_files)


@dataclass
class BatchJob:
    """Represents a batch processing job for a single VTT file.

    Attributes:
        vtt_file: Path to the VTT file
        title: Meeting title for the minutes
        output_file: Path for the output markdown file
        enabled: Whether this job should be processed
        intermediate_file: Optional path for intermediate file
    """

    vtt_file: Path
    title: str
    output_file: Path
    enabled: bool = True
    intermediate_file: Path | None = None

    def __post_init__(self) -> None:
        """Validate job configuration after initialization."""
        if not self.vtt_file.exists():
            raise FileNotFoundError(f"VTT file not found: {self.vtt_file}")

        if not self.vtt_file.suffix.lower() == ".vtt":
            raise ValueError(f"File is not a VTT file: {self.vtt_file}")

        if not self.title.strip():
            raise ValueError("Title cannot be empty")

    @classmethod
    def from_vtt_file(
        cls, vtt_file: Path, output_dir: Path | None = None, title: str | None = None
    ) -> "BatchJob":
        """Create a BatchJob with default values from a VTT file.

        Args:
            vtt_file: Path to the VTT file
            output_dir: Directory for output file (default: same as VTT file)
            title: Meeting title (default: filename without extension)

        Returns:
            BatchJob with default configuration
        """
        default_title = title or _generate_default_title(vtt_file)
        default_output_dir = output_dir or vtt_file.parent
        default_output_file = default_output_dir / f"{vtt_file.stem}.md"

        return cls(
            vtt_file=vtt_file, title=default_title, output_file=default_output_file
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary representation.

        Returns:
            Dictionary representation of the job
        """
        return {
            "vtt_file": str(self.vtt_file),
            "title": self.title,
            "output_file": str(self.output_file),
            "enabled": self.enabled,
            "intermediate_file": str(self.intermediate_file)
            if self.intermediate_file
            else None,
        }


def _generate_default_title(vtt_file: Path) -> str:
    """Generate a default title from VTT filename.

    Args:
        vtt_file: Path to the VTT file

    Returns:
        Default title based on filename
    """
    # Remove extension and replace underscores/hyphens with spaces
    title = vtt_file.stem.replace("_", " ").replace("-", " ")
    # Capitalize first letter of each word
    return title.title()


class BatchJobCollection:
    """Collection of batch jobs with validation and management."""

    def __init__(self, jobs: list[BatchJob] | None = None) -> None:
        """Initialize job collection.

        Args:
            jobs: Initial list of jobs
        """
        self._jobs = jobs or []

    def add_job(self, job: BatchJob) -> None:
        """Add a job to the collection.

        Args:
            job: BatchJob to add

        Raises:
            ValueError: If job with same VTT file already exists
        """
        if any(existing.vtt_file == job.vtt_file for existing in self._jobs):
            raise ValueError(f"Job for {job.vtt_file} already exists")

        self._jobs.append(job)

    def remove_job(self, vtt_file: Path) -> bool:
        """Remove job by VTT file path.

        Args:
            vtt_file: Path to the VTT file

        Returns:
            True if job was removed, False if not found
        """
        for i, job in enumerate(self._jobs):
            if job.vtt_file == vtt_file:
                del self._jobs[i]
                return True
        return False

    def get_enabled_jobs(self) -> list[BatchJob]:
        """Get list of enabled jobs.

        Returns:
            List of enabled BatchJob instances
        """
        return [job for job in self._jobs if job.enabled]

    def get_all_jobs(self) -> list[BatchJob]:
        """Get all jobs in the collection.

        Returns:
            List of all BatchJob instances
        """
        return self._jobs.copy()

    def __len__(self) -> int:
        """Get number of jobs in collection."""
        return len(self._jobs)

    def __iter__(self):
        """Iterate over jobs in collection."""
        return iter(self._jobs)

    def validate_output_conflicts(self) -> list[tuple[BatchJob, BatchJob]]:
        """Check for output file conflicts between enabled jobs.

        Returns:
            List of job pairs with conflicting output files
        """
        conflicts: list[tuple[BatchJob, BatchJob]] = []
        enabled_jobs = self.get_enabled_jobs()

        for i, job1 in enumerate(enabled_jobs):
            for job2 in enabled_jobs[i + 1 :]:
                if job1.output_file == job2.output_file:
                    conflicts.append((job1, job2))

        return conflicts


def display_vtt_files_table(
    vtt_files: list[Path], base_directory: Path, console: Console | None = None
) -> None:
    """Display VTT files in a formatted table.

    Args:
        vtt_files: List of VTT file paths
        base_directory: Base directory for relative path display
        console: Rich console instance (creates new if None)
    """
    if console is None:
        console = Console()

    if not vtt_files:
        console.print(
            "[yellow]指定されたディレクトリにVTTファイルが見つかりませんでした。[/yellow]"
        )
        return

    # Create table
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("No.", style="dim", width=4, justify="right")
    table.add_column("ファイル名", style="cyan", min_width=30)
    table.add_column("サイズ", justify="right", width=10)
    table.add_column("更新日時", width=20)
    table.add_column("相対パス", style="dim")

    # Add rows
    for i, vtt_file in enumerate(vtt_files, 1):
        try:
            # Get file stats
            stat = vtt_file.stat()
            size = _format_file_size(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            # Get relative path from base directory
            try:
                rel_path = vtt_file.relative_to(base_directory)
                rel_path_str = (
                    str(rel_path.parent) if rel_path.parent != Path(".") else ""
                )
            except ValueError:
                # File is outside base directory
                rel_path_str = str(vtt_file.parent)

            table.add_row(str(i), vtt_file.name, size, mtime, rel_path_str)
        except (OSError, PermissionError):
            # Handle files that can't be accessed
            table.add_row(str(i), vtt_file.name, "?", "?", str(vtt_file.parent))

    # Display header and table
    console.print()
    console.print(
        Panel.fit(
            Text(f"見つかったVTTファイル: {len(vtt_files)}件", style="bold green"),
            style="green",
        )
    )
    console.print(table)
    console.print()


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string
    """
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def display_batch_summary(
    jobs: list[BatchJob], base_directory: Path, console: Console | None = None
) -> None:
    """Display summary of batch jobs before processing.

    Args:
        jobs: List of batch jobs
        base_directory: Base directory for relative paths
        console: Rich console instance (creates new if None)
    """
    if console is None:
        console = Console()

    enabled_jobs = [job for job in jobs if job.enabled]
    disabled_jobs = [job for job in jobs if not job.enabled]

    # Summary header
    console.print()
    console.print(
        Panel.fit(Text("バッチ処理サマリー", style="bold blue"), style="blue")
    )

    # Statistics
    console.print(f"[bold]処理対象:[/bold] {len(enabled_jobs)}件")
    console.print(f"[bold]スキップ:[/bold] {len(disabled_jobs)}件")
    console.print(f"[bold]合計:[/bold] {len(jobs)}件")
    console.print()

    if not enabled_jobs:
        console.print("[yellow]処理対象のファイルがありません。[/yellow]")
        return

    # Enabled jobs table
    table = Table(show_header=True, header_style="bold green")
    table.add_column("No.", style="dim", width=4, justify="right")
    table.add_column("VTTファイル", style="cyan", min_width=25)
    table.add_column("タイトル", style="yellow", min_width=20)
    table.add_column("出力ファイル", style="green", min_width=25)

    for i, job in enumerate(enabled_jobs, 1):
        try:
            vtt_rel = job.vtt_file.relative_to(base_directory)
            output_rel = job.output_file.relative_to(base_directory)
        except ValueError:
            vtt_rel = job.vtt_file
            output_rel = job.output_file

        table.add_row(str(i), str(vtt_rel), job.title, str(output_rel))

    console.print("[bold green]処理対象ファイル:[/bold green]")
    console.print(table)
    console.print()


def interactive_job_configuration(
    vtt_files: list[Path],
    base_directory: Path,
    output_directory: Path | None = None,
    console: Console | None = None,
) -> list[BatchJob]:
    """Interactively configure batch jobs for VTT files.

    Args:
        vtt_files: List of VTT files to configure
        base_directory: Base directory for relative paths
        output_directory: Default output directory (default: same as VTT file)
        console: Rich console instance (creates new if None)

    Returns:
        List of configured BatchJob instances
    """
    if console is None:
        console = Console()

    if not vtt_files:
        console.print("[yellow]設定するVTTファイルがありません。[/yellow]")
        return []

    jobs: list[BatchJob] = []

    console.print()
    console.print(
        Panel.fit(
            Text("バッチ処理設定", style="bold blue"),
            style="blue",
        )
    )
    console.print(
        "[dim]各ファイルについて設定を行います。Enterキーでデフォルト値を使用できます。[/dim]"
    )
    console.print()

    for i, vtt_file in enumerate(vtt_files, 1):
        try:
            rel_path = vtt_file.relative_to(base_directory)
        except ValueError:
            rel_path = vtt_file

        console.print(
            f"[bold cyan]ファイル {i}/{len(vtt_files)}: {rel_path}[/bold cyan]"
        )

        # Generate defaults
        default_title = _generate_default_title(vtt_file)
        if output_directory:
            default_output = output_directory / f"{vtt_file.stem}.md"
        else:
            default_output = vtt_file.parent / f"{vtt_file.stem}.md"

        try:
            default_output_rel = default_output.relative_to(base_directory)
        except ValueError:
            default_output_rel = default_output

        # Interactive input
        title = _prompt_for_title(default_title, console)
        output_file = _prompt_for_output_path(
            default_output, default_output_rel, base_directory, console
        )
        process_file = _prompt_for_processing_confirmation(console)

        if process_file:
            job = BatchJob(
                vtt_file=vtt_file, title=title, output_file=output_file, enabled=True
            )
            jobs.append(job)
            console.print("[green]✓ 処理対象に追加しました[/green]")
        else:
            console.print("[yellow]✗ スキップします[/yellow]")

        console.print()

    return jobs


def _prompt_for_title(default_title: str, console: Console) -> str:
    """Prompt user for meeting title with default value.

    Args:
        default_title: Default title to show
        console: Rich console instance

    Returns:
        User-provided or default title
    """
    try:
        title = Prompt.ask(
            f"[yellow]タイトル[/yellow] [dim]\\[{default_title}][/dim]",
            default=default_title,
            console=console,
        ).strip()

        if not title:
            return default_title
        return title
    except (KeyboardInterrupt, EOFError):
        console.print("[yellow]\\n中断されました。デフォルト値を使用します。[/yellow]")
        return default_title


def _prompt_for_output_path(
    default_output: Path,
    default_output_rel: Path,
    base_directory: Path,
    console: Console,
) -> Path:
    """Prompt user for output file path with default value.

    Args:
        default_output: Default output path (absolute)
        default_output_rel: Default output path (relative to base)
        base_directory: Base directory for relative path resolution
        console: Rich console instance

    Returns:
        User-provided or default output path
    """
    try:
        output_str = Prompt.ask(
            f"[yellow]出力先[/yellow] [dim]\\[{default_output_rel}][/dim]",
            default=str(default_output_rel),
            console=console,
        ).strip()

        if not output_str:
            return default_output

        # Convert to Path and handle relative paths
        output_path = Path(output_str)
        if not output_path.is_absolute():
            output_path = base_directory / output_path

        return output_path
    except (KeyboardInterrupt, EOFError):
        console.print("[yellow]\\n中断されました。デフォルト値を使用します。[/yellow]")
        return default_output


def _prompt_for_processing_confirmation(console: Console) -> bool:
    """Prompt user to confirm processing this file.

    Args:
        console: Rich console instance

    Returns:
        True if file should be processed, False otherwise
    """
    try:
        return Confirm.ask("[yellow]処理する?[/yellow]", default=True, console=console)
    except (KeyboardInterrupt, EOFError):
        console.print("[yellow]\\n中断されました。処理をスキップします。[/yellow]")
        return False


def _display_conflicts(
    conflicts: list[tuple[BatchJob, BatchJob]], console: Console
) -> None:
    """Display output file conflicts."""
    if not conflicts:
        return

    console.print("[red bold]⚠ 出力ファイルの競合が検出されました:[/red bold]")
    for job1, job2 in conflicts:
        console.print(f"  [red]• {job1.output_file}[/red]")
        console.print(f"    - {job1.vtt_file.name} ({job1.title})")
        console.print(f"    - {job2.vtt_file.name} ({job2.title})")
    console.print()


def _get_user_choice(
    conflicts: list[tuple[BatchJob, BatchJob]], console: Console
) -> str:
    """Get user choice for job confirmation."""
    console.print("[bold]このまま処理を開始しますか？[/bold]")
    console.print("  [green]y[/green] - 処理開始")
    console.print("  [yellow]e[/yellow] - 設定を編集")
    console.print("  [red]n[/red] - キャンセル")

    return Prompt.ask(
        "選択",
        choices=["y", "e", "n"],
        default="y" if not conflicts else "e",
        console=console,
    ).lower()


def _handle_proceed_choice(
    conflicts: list[tuple[BatchJob, BatchJob]], console: Console
) -> bool:
    """Handle proceed choice with conflict confirmation."""
    if not conflicts:
        return True

    return Confirm.ask(
        "[red]競合があります。本当に続行しますか？[/red]",
        default=False,
        console=console,
    )


def review_and_confirm_jobs(
    jobs: list[BatchJob], base_directory: Path, console: Console | None = None
) -> tuple[bool, list[BatchJob]]:
    """Review configured jobs and allow modifications before processing.

    Args:
        jobs: List of configured jobs
        base_directory: Base directory for relative paths
        console: Rich console instance (creates new if None)

    Returns:
        Tuple of (proceed_with_processing, final_jobs_list)
    """
    if console is None:
        console = Console()

    if not jobs:
        console.print("[yellow]処理対象のファイルがありません。[/yellow]")
        return False, []

    while True:
        # Display current configuration and check conflicts
        display_batch_summary(jobs, base_directory, console)
        collection = BatchJobCollection(jobs)
        conflicts = collection.validate_output_conflicts()
        _display_conflicts(conflicts, console)

        try:
            choice = _get_user_choice(conflicts, console)

            if choice == "y":
                if _handle_proceed_choice(conflicts, console):
                    return True, jobs
                # If user declined to proceed with conflicts, continue the loop
            elif choice == "e":
                jobs = _edit_job_configuration(jobs, base_directory, console)
                if not jobs:  # User cancelled editing
                    return False, []
            else:  # choice == "n"
                return False, []

        except (KeyboardInterrupt, EOFError):
            console.print("[yellow]\\n処理をキャンセルしました。[/yellow]")
            return False, []


def _edit_job_configuration(
    jobs: list[BatchJob], base_directory: Path, console: Console
) -> list[BatchJob]:
    """Allow user to edit job configuration.

    Args:
        jobs: Current list of jobs
        base_directory: Base directory for relative paths
        console: Rich console instance

    Returns:
        Modified list of jobs (empty list if cancelled)
    """
    console.print()
    console.print("[bold blue]設定編集モード[/bold blue]")
    console.print("[dim]編集したいファイルの番号を入力してください（0で完了）[/dim]")
    console.print()

    while True:
        # Display numbered list
        for i, job in enumerate(jobs, 1):
            status = "[green]有効[/green]" if job.enabled else "[red]無効[/red]"
            try:
                rel_path = job.vtt_file.relative_to(base_directory)
            except ValueError:
                rel_path = job.vtt_file

            console.print(f"  [cyan]{i:2d}[/cyan]. {rel_path} - {job.title} ({status})")

        console.print()

        try:
            choice = Prompt.ask(
                "編集するファイル番号 (0で完了)", default="0", console=console
            )

            if choice == "0":
                break

            try:
                index = int(choice) - 1
                if 0 <= index < len(jobs):
                    jobs[index] = _edit_single_job(jobs[index], base_directory, console)
                else:
                    console.print(f"[red]無効な番号です: {choice}[/red]")
            except ValueError:
                console.print(f"[red]数値を入力してください: {choice}[/red]")

        except (KeyboardInterrupt, EOFError):
            console.print("[yellow]\\n編集をキャンセルしました。[/yellow]")
            return []

    return jobs


def _edit_single_job(job: BatchJob, base_directory: Path, console: Console) -> BatchJob:
    """Edit a single job configuration.

    Args:
        job: Job to edit
        base_directory: Base directory for relative paths
        console: Rich console instance

    Returns:
        Modified job
    """
    try:
        rel_path = job.vtt_file.relative_to(base_directory)
    except ValueError:
        rel_path = job.vtt_file

    console.print(f"\\n[bold cyan]編集中: {rel_path}[/bold cyan]")

    # Edit title
    new_title = _prompt_for_title(job.title, console)

    # Edit output path
    try:
        current_output_rel = job.output_file.relative_to(base_directory)
    except ValueError:
        current_output_rel = job.output_file

    new_output = _prompt_for_output_path(
        job.output_file, current_output_rel, base_directory, console
    )

    # Edit enabled status
    new_enabled = Confirm.ask(
        "[yellow]このファイルを処理する?[/yellow]", default=job.enabled, console=console
    )

    # Return updated job
    return BatchJob(
        vtt_file=job.vtt_file,
        title=new_title,
        output_file=new_output,
        enabled=new_enabled,
        intermediate_file=job.intermediate_file,
    )


class BatchProcessor:
    """Processes multiple VTT files in batch mode."""

    def __init__(self, config: ProcessingConfig) -> None:
        """Initialize batch processor.

        Args:
            config: Processing configuration
        """
        self.config = config
        self.console = Console()

    def process_batch(self, jobs: list[BatchJob]) -> list[ProcessingResult]:
        """Process multiple VTT files.

        Args:
            jobs: List of batch jobs to process

        Returns:
            List of processing results
        """
        enabled_jobs = [job for job in jobs if job.enabled]
        if not enabled_jobs:
            self.console.print("[yellow]処理対象のファイルがありません。[/yellow]")
            return []

        results: list[ProcessingResult] = []
        processor = VTTFileProcessor(self.config)

        self.console.print()
        self.console.print(
            Panel.fit(
                Text(f"バッチ処理開始: {len(enabled_jobs)}件", style="bold green"),
                style="green",
            )
        )

        # Process each job with overall progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            overall_task = progress.add_task("バッチ処理中...", total=len(enabled_jobs))

            for i, job in enumerate(enabled_jobs, 1):
                try:
                    rel_path = job.vtt_file.relative_to(Path.cwd())
                except ValueError:
                    rel_path = job.vtt_file

                progress.update(
                    overall_task,
                    description=f"処理中 ({i}/{len(enabled_jobs)}): {rel_path.name}",
                )

                # Check if this is for chat prompt generation
                chat_prompt_file = None
                if job.output_file.suffix == ".txt":
                    chat_prompt_file = job.output_file
                    # Use markdown extension for actual output
                    actual_output = job.output_file.with_suffix(".md")
                else:
                    actual_output = job.output_file

                # Process single file
                result = processor.process_file(
                    input_file=job.vtt_file,
                    output_file=actual_output,
                    title=job.title,
                    intermediate_file=job.intermediate_file,
                    chat_prompt_file=chat_prompt_file,
                )

                results.append(result)

                # Handle processing errors
                if not result.success:
                    if self._should_continue_on_error(result, i, len(enabled_jobs)):
                        progress.advance(overall_task)
                        continue
                    else:
                        # User chose to stop processing
                        break

                progress.advance(overall_task)

        # Display final summary
        self._display_batch_results(results, enabled_jobs)
        return results

    def _should_continue_on_error(
        self, result: ProcessingResult, current: int, total: int
    ) -> bool:
        """Ask user whether to continue processing after an error.

        Args:
            result: Failed processing result
            current: Current file index
            total: Total number of files

        Returns:
            True to continue, False to stop
        """
        try:
            rel_path = result.input_file.relative_to(Path.cwd())
        except ValueError:
            rel_path = result.input_file

        self.console.print()
        self.console.print("[red bold]エラーが発生しました:[/red bold]")
        self.console.print(f"[red]ファイル: {rel_path}[/red]")
        self.console.print(f"[red]エラー: {result.error}[/red]")
        self.console.print()

        if current < total:
            remaining = total - current
            return Confirm.ask(
                f"[yellow]残り{remaining}件の処理を続行しますか？[/yellow]",
                default=True,
                console=self.console,
            )
        return False

    def _display_batch_results(
        self, results: list[ProcessingResult], jobs: list[BatchJob]
    ) -> None:
        """Display summary of batch processing results.

        Args:
            results: List of processing results
            jobs: List of processed jobs
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        self.console.print()
        self.console.print(
            Panel.fit(Text("バッチ処理完了", style="bold blue"), style="blue")
        )

        # Summary statistics
        self.console.print(f"[bold]処理対象:[/bold] {len(jobs)}件")
        self.console.print(f"[bold green]成功:[/bold green] {len(successful)}件")
        if failed:
            self.console.print(f"[bold red]失敗:[/bold red] {len(failed)}件")
        self.console.print()

        # Success table
        if successful:
            self._display_success_table(successful)

        # Failure table
        if failed:
            self._display_failure_table(failed)

    def _display_success_table(self, results: list[ProcessingResult]) -> None:
        """Display table of successful processing results."""
        table = Table(show_header=True, header_style="bold green")
        table.add_column("No.", style="dim", width=4, justify="right")
        table.add_column("入力ファイル", style="cyan", min_width=25)
        table.add_column("出力ファイル", style="green", min_width=25)

        for i, result in enumerate(results, 1):
            try:
                input_rel = result.input_file.relative_to(Path.cwd())
                output_rel = (
                    result.output_file.relative_to(Path.cwd())
                    if result.output_file
                    else "?"
                )
            except ValueError:
                input_rel = result.input_file
                output_rel = result.output_file or "?"

            table.add_row(str(i), str(input_rel), str(output_rel))

        self.console.print("[bold green]成功したファイル:[/bold green]")
        self.console.print(table)
        self.console.print()

    def _display_failure_table(self, results: list[ProcessingResult]) -> None:
        """Display table of failed processing results."""
        table = Table(show_header=True, header_style="bold red")
        table.add_column("No.", style="dim", width=4, justify="right")
        table.add_column("入力ファイル", style="cyan", min_width=25)
        table.add_column("エラー", style="red", min_width=30)

        for i, result in enumerate(results, 1):
            try:
                input_rel = result.input_file.relative_to(Path.cwd())
            except ValueError:
                input_rel = result.input_file

            # Truncate long error messages
            error_msg = result.error or "不明なエラー"
            if len(error_msg) > 50:
                error_msg = error_msg[:47] + "..."

            table.add_row(str(i), str(input_rel), error_msg)

        self.console.print("[bold red]失敗したファイル:[/bold red]")
        self.console.print(table)
        self.console.print()

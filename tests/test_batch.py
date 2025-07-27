"""Tests for batch processing functionality."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from vtt2minutes.batch import (
    BatchJob,
    BatchJobCollection,
    BatchProcessor,
    _format_file_size,
    _generate_default_title,
    _prompt_for_output_path,
    _prompt_for_processing_confirmation,
    _prompt_for_title,
    display_batch_summary,
    display_vtt_files_table,
    interactive_job_configuration,
    review_and_confirm_jobs,
    scan_vtt_files,
)
from vtt2minutes.processor import ProcessingConfig, ProcessingResult


class TestScanVttFiles:
    """Tests for scan_vtt_files function."""

    def test_scan_empty_directory(self) -> None:
        """Test scanning an empty directory."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            result = scan_vtt_files(directory)
            assert result == []

    def test_scan_directory_with_vtt_files(self) -> None:
        """Test scanning directory with VTT files."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create test VTT files
            file1 = directory / "meeting1.vtt"
            file2 = directory / "meeting2.vtt"
            file3 = directory / "document.txt"  # Non-VTT file

            file1.touch()
            file2.touch()
            file3.touch()

            result = scan_vtt_files(directory)
            assert len(result) == 2
            assert file1 in result
            assert file2 in result
            assert file3 not in result

    def test_scan_recursive(self) -> None:
        """Test recursive scanning."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create nested structure
            subdir = directory / "subdir"
            subdir.mkdir()

            file1 = directory / "root.vtt"
            file2 = subdir / "nested.vtt"

            file1.touch()
            file2.touch()

            result = scan_vtt_files(directory, recursive=True)
            assert len(result) == 2
            assert file1 in result
            assert file2 in result

    def test_scan_non_recursive(self) -> None:
        """Test non-recursive scanning."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create nested structure
            subdir = directory / "subdir"
            subdir.mkdir()

            file1 = directory / "root.vtt"
            file2 = subdir / "nested.vtt"

            file1.touch()
            file2.touch()

            result = scan_vtt_files(directory, recursive=False)
            assert len(result) == 1
            assert file1 in result
            assert file2 not in result

    def test_skip_hidden_files(self) -> None:
        """Test that hidden files are skipped."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create hidden and normal files
            visible_file = directory / "meeting.vtt"
            hidden_file = directory / ".hidden.vtt"

            # Create hidden directory with file
            hidden_dir = directory / ".hidden_dir"
            hidden_dir.mkdir()
            file_in_hidden_dir = hidden_dir / "meeting.vtt"

            visible_file.touch()
            hidden_file.touch()
            file_in_hidden_dir.touch()

            result = scan_vtt_files(directory)
            assert len(result) == 1
            assert visible_file in result
            assert hidden_file not in result
            assert file_in_hidden_dir not in result

    def test_sorted_output(self) -> None:
        """Test that output is sorted by filename."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create files in non-alphabetical order
            file_c = directory / "c_meeting.vtt"
            file_a = directory / "a_meeting.vtt"
            file_b = directory / "b_meeting.vtt"

            file_c.touch()
            file_a.touch()
            file_b.touch()

            result = scan_vtt_files(directory)
            assert result == [file_a, file_b, file_c]

    def test_directory_not_found(self) -> None:
        """Test error when directory does not exist."""
        non_existent = Path("/non/existent/directory")
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            scan_vtt_files(non_existent)

    def test_path_is_not_directory(self) -> None:
        """Test error when path is not a directory."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            file_path = directory / "test.txt"
            file_path.touch()

            with pytest.raises(NotADirectoryError, match="Path is not a directory"):
                scan_vtt_files(file_path)


class TestBatchJob:
    """Tests for BatchJob class."""

    def test_create_batch_job(self) -> None:
        """Test creating a valid BatchJob."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            output_file = directory / "meeting.md"
            vtt_file.touch()

            job = BatchJob(
                vtt_file=vtt_file, title="Test Meeting", output_file=output_file
            )

            assert job.vtt_file == vtt_file
            assert job.title == "Test Meeting"
            assert job.output_file == output_file
            assert job.enabled is True
            assert job.intermediate_file is None

    def test_batch_job_validation_missing_file(self) -> None:
        """Test validation fails for missing VTT file."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "missing.vtt"
            output_file = directory / "output.md"

            with pytest.raises(FileNotFoundError, match="VTT file not found"):
                BatchJob(vtt_file=vtt_file, title="Test", output_file=output_file)

    def test_batch_job_validation_wrong_extension(self) -> None:
        """Test validation fails for non-VTT file."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            txt_file = directory / "document.txt"
            output_file = directory / "output.md"
            txt_file.touch()

            with pytest.raises(ValueError, match="File is not a VTT file"):
                BatchJob(vtt_file=txt_file, title="Test", output_file=output_file)

    def test_batch_job_validation_empty_title(self) -> None:
        """Test validation fails for empty title."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            output_file = directory / "meeting.md"
            vtt_file.touch()

            with pytest.raises(ValueError, match="Title cannot be empty"):
                BatchJob(
                    vtt_file=vtt_file,
                    title="   ",  # Only whitespace
                    output_file=output_file,
                )

    def test_from_vtt_file_defaults(self) -> None:
        """Test creating BatchJob with default values."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "project_meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)

            assert job.vtt_file == vtt_file
            assert job.title == "Project Meeting"
            assert job.output_file == directory / "project_meeting.md"
            assert job.enabled is True

    def test_from_vtt_file_custom_values(self) -> None:
        """Test creating BatchJob with custom values."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            output_dir = directory / "output"
            output_dir.mkdir()
            vtt_file = directory / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(
                vtt_file, output_dir=output_dir, title="Custom Title"
            )

            assert job.vtt_file == vtt_file
            assert job.title == "Custom Title"
            assert job.output_file == output_dir / "meeting.md"

    def test_to_dict(self) -> None:
        """Test converting job to dictionary."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            output_file = directory / "meeting.md"
            intermediate_file = directory / "meeting.preprocessed.md"
            vtt_file.touch()

            job = BatchJob(
                vtt_file=vtt_file,
                title="Test Meeting",
                output_file=output_file,
                enabled=False,
                intermediate_file=intermediate_file,
            )

            result = job.to_dict()

            assert result["vtt_file"] == str(vtt_file)
            assert result["title"] == "Test Meeting"
            assert result["output_file"] == str(output_file)
            assert result["enabled"] is False
            assert result["intermediate_file"] == str(intermediate_file)


class TestGenerateDefaultTitle:
    """Tests for _generate_default_title function."""

    def test_simple_filename(self) -> None:
        """Test generating title from simple filename."""
        vtt_file = Path("meeting.vtt")
        title = _generate_default_title(vtt_file)
        assert title == "Meeting"

    def test_underscore_filename(self) -> None:
        """Test generating title from filename with underscores."""
        vtt_file = Path("project_planning_meeting.vtt")
        title = _generate_default_title(vtt_file)
        assert title == "Project Planning Meeting"

    def test_hyphen_filename(self) -> None:
        """Test generating title from filename with hyphens."""
        vtt_file = Path("team-standup-daily.vtt")
        title = _generate_default_title(vtt_file)
        assert title == "Team Standup Daily"

    def test_mixed_separators(self) -> None:
        """Test generating title from filename with mixed separators."""
        vtt_file = Path("project_review-meeting_notes.vtt")
        title = _generate_default_title(vtt_file)
        assert title == "Project Review Meeting Notes"


class TestBatchJobCollection:
    """Tests for BatchJobCollection class."""

    def test_empty_collection(self) -> None:
        """Test creating empty job collection."""
        collection = BatchJobCollection()
        assert len(collection) == 0
        assert collection.get_all_jobs() == []
        assert collection.get_enabled_jobs() == []

    def test_add_job(self) -> None:
        """Test adding jobs to collection."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            collection = BatchJobCollection()
            collection.add_job(job)

            assert len(collection) == 1
            assert job in collection.get_all_jobs()

    def test_add_duplicate_job(self) -> None:
        """Test adding duplicate job raises error."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            vtt_file.touch()

            job1 = BatchJob.from_vtt_file(vtt_file)
            job2 = BatchJob.from_vtt_file(vtt_file)

            collection = BatchJobCollection()
            collection.add_job(job1)

            with pytest.raises(ValueError, match="Job for .* already exists"):
                collection.add_job(job2)

    def test_remove_job(self) -> None:
        """Test removing job from collection."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            collection = BatchJobCollection([job])

            assert len(collection) == 1
            result = collection.remove_job(vtt_file)
            assert result is True
            assert len(collection) == 0

    def test_remove_nonexistent_job(self) -> None:
        """Test removing non-existent job returns False."""
        collection = BatchJobCollection()
        result = collection.remove_job(Path("nonexistent.vtt"))
        assert result is False

    def test_get_enabled_jobs(self) -> None:
        """Test getting only enabled jobs."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            vtt_file1 = directory / "meeting1.vtt"
            vtt_file2 = directory / "meeting2.vtt"
            vtt_file1.touch()
            vtt_file2.touch()

            job1 = BatchJob.from_vtt_file(vtt_file1)
            job2 = BatchJob.from_vtt_file(vtt_file2)
            job2.enabled = False

            collection = BatchJobCollection([job1, job2])

            enabled = collection.get_enabled_jobs()
            assert len(enabled) == 1
            assert job1 in enabled
            assert job2 not in enabled

    def test_iteration(self) -> None:
        """Test iterating over collection."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            collection = BatchJobCollection([job])

            jobs = list(collection)
            assert len(jobs) == 1
            assert jobs[0] == job

    def test_validate_output_conflicts(self) -> None:
        """Test detecting output file conflicts."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            vtt_file1 = directory / "meeting1.vtt"
            vtt_file2 = directory / "meeting2.vtt"
            vtt_file1.touch()
            vtt_file2.touch()

            # Create jobs with same output file
            output_file = directory / "same_output.md"
            job1 = BatchJob(
                vtt_file=vtt_file1, title="Meeting 1", output_file=output_file
            )
            job2 = BatchJob(
                vtt_file=vtt_file2, title="Meeting 2", output_file=output_file
            )

            collection = BatchJobCollection([job1, job2])
            conflicts = collection.validate_output_conflicts()

            assert len(conflicts) == 1
            assert (job1, job2) in conflicts or (job2, job1) in conflicts

    def test_no_conflicts_with_disabled_jobs(self) -> None:
        """Test that disabled jobs don't cause conflicts."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            vtt_file1 = directory / "meeting1.vtt"
            vtt_file2 = directory / "meeting2.vtt"
            vtt_file1.touch()
            vtt_file2.touch()

            # Create jobs with same output file, but one disabled
            output_file = directory / "same_output.md"
            job1 = BatchJob(
                vtt_file=vtt_file1, title="Meeting 1", output_file=output_file
            )
            job2 = BatchJob(
                vtt_file=vtt_file2,
                title="Meeting 2",
                output_file=output_file,
                enabled=False,
            )

            collection = BatchJobCollection([job1, job2])
            conflicts = collection.validate_output_conflicts()

            assert len(conflicts) == 0


class TestFormatFileSize:
    """Tests for _format_file_size function."""

    def test_bytes(self) -> None:
        """Test formatting bytes."""
        assert _format_file_size(100) == "100B"
        assert _format_file_size(1023) == "1023B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert _format_file_size(1024) == "1.0KB"
        assert _format_file_size(1536) == "1.5KB"
        assert _format_file_size(1048575) == "1024.0KB"

    def test_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert _format_file_size(1048576) == "1.0MB"
        assert _format_file_size(2097152) == "2.0MB"
        assert _format_file_size(1572864) == "1.5MB"


class TestDisplayVttFilesTable:
    """Tests for display_vtt_files_table function."""

    def test_empty_file_list(self) -> None:
        """Test displaying empty file list."""
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        display_vtt_files_table([], Path("/test"), console)

        # Need to access StringIO directly since console.file might not have getvalue
        output = string_io.getvalue()
        assert "指定されたディレクトリにVTTファイルが見つかりませんでした" in output

    def test_display_files(self) -> None:
        """Test displaying VTT files."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create test files
            file1 = directory / "meeting1.vtt"
            file2 = directory / "subdir" / "meeting2.vtt"
            file2.parent.mkdir()

            file1.write_text("test content")
            file2.write_text("longer test content")

            string_io = StringIO()
            console = Console(file=string_io, width=120)
            display_vtt_files_table([file1, file2], directory, console)

            output = string_io.getvalue()
            assert "見つかったVTTファイル: 2件" in output
            assert "meeting1.vtt" in output
            assert "meeting2.vtt" in output
            assert "subdir" in output

    def test_file_access_error(self) -> None:
        """Test handling files that can't be accessed."""
        # Create a file path that doesn't exist
        non_existent = Path("/non/existent/file.vtt")

        string_io = StringIO()
        console = Console(file=string_io, width=80)
        display_vtt_files_table([non_existent], Path("/"), console)

        output = string_io.getvalue()
        assert "file.vtt" in output
        # Should show "?" for size and mtime when file can't be accessed


class TestDisplayBatchSummary:
    """Tests for display_batch_summary function."""

    def test_empty_jobs(self) -> None:
        """Test displaying summary with no jobs."""
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        display_batch_summary([], Path("/test"), console)

        output = string_io.getvalue()
        assert "バッチ処理サマリー" in output
        assert "処理対象: 0件" in output
        assert "処理対象のファイルがありません" in output

    def test_mixed_jobs(self) -> None:
        """Test displaying summary with enabled and disabled jobs."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            # Create test files
            file1 = directory / "meeting1.vtt"
            file2 = directory / "meeting2.vtt"
            file1.touch()
            file2.touch()

            # Create jobs
            job1 = BatchJob.from_vtt_file(file1)
            job2 = BatchJob.from_vtt_file(file2)
            job2.enabled = False

            string_io = StringIO()
            console = Console(file=string_io, width=120)
            display_batch_summary([job1, job2], directory, console)

            output = string_io.getvalue()
            assert "バッチ処理サマリー" in output
            assert "処理対象: 1件" in output
            assert "スキップ: 1件" in output
            assert "合計: 2件" in output
            assert "Meeting1" in output  # Title from job1
            assert "処理対象ファイル" in output

    def test_all_disabled_jobs(self) -> None:
        """Test displaying summary with all jobs disabled."""
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            vtt_file = directory / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            job.enabled = False

            string_io = StringIO()
            console = Console(file=string_io, width=80)
            display_batch_summary([job], directory, console)

            output = string_io.getvalue()
            assert "処理対象: 0件" in output
            assert "スキップ: 1件" in output
            assert "処理対象のファイルがありません" in output


class TestInteractiveFunctions:
    """Tests for interactive input functions."""

    @patch("vtt2minutes.batch.Prompt.ask")
    def test_prompt_for_title_default(self, mock_prompt) -> None:
        """Test prompting for title with default value."""
        mock_prompt.return_value = ""
        console = Console()
        result = _prompt_for_title("Default Title", console)
        assert result == "Default Title"

    @patch("vtt2minutes.batch.Prompt.ask")
    def test_prompt_for_title_custom(self, mock_prompt) -> None:
        """Test prompting for title with custom input."""
        mock_prompt.return_value = "Custom Title"
        console = Console()
        result = _prompt_for_title("Default Title", console)
        assert result == "Custom Title"

    @patch("vtt2minutes.batch.Prompt.ask")
    def test_prompt_for_title_keyboard_interrupt(self, mock_prompt) -> None:
        """Test handling keyboard interrupt in title prompt."""
        mock_prompt.side_effect = KeyboardInterrupt()
        console = Console()
        result = _prompt_for_title("Default Title", console)
        assert result == "Default Title"

    @patch("vtt2minutes.batch.Prompt.ask")
    def test_prompt_for_output_path_default(self, mock_prompt) -> None:
        """Test prompting for output path with default value."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_output = base_dir / "output.md"
            default_rel = Path("output.md")

            mock_prompt.return_value = ""
            console = Console()

            result = _prompt_for_output_path(
                default_output, default_rel, base_dir, console
            )
            assert result == default_output

    @patch("vtt2minutes.batch.Prompt.ask")
    def test_prompt_for_output_path_relative(self, mock_prompt) -> None:
        """Test prompting for output path with relative input."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            default_output = base_dir / "default.md"
            default_rel = Path("default.md")

            mock_prompt.return_value = "custom/output.md"
            console = Console()

            result = _prompt_for_output_path(
                default_output, default_rel, base_dir, console
            )
            assert result == base_dir / "custom" / "output.md"

    @patch("vtt2minutes.batch.Confirm.ask")
    def test_prompt_for_processing_confirmation_yes(self, mock_confirm) -> None:
        """Test processing confirmation returning True."""
        mock_confirm.return_value = True
        console = Console()
        result = _prompt_for_processing_confirmation(console)
        assert result is True

    @patch("vtt2minutes.batch.Confirm.ask")
    def test_prompt_for_processing_confirmation_no(self, mock_confirm) -> None:
        """Test processing confirmation returning False."""
        mock_confirm.return_value = False
        console = Console()
        result = _prompt_for_processing_confirmation(console)
        assert result is False

    @patch("vtt2minutes.batch.Confirm.ask")
    def test_prompt_for_processing_confirmation_interrupt(self, mock_confirm) -> None:
        """Test handling interrupt in processing confirmation."""
        mock_confirm.side_effect = KeyboardInterrupt()
        console = Console()
        result = _prompt_for_processing_confirmation(console)
        assert result is False


class TestInteractiveJobConfiguration:
    """Tests for interactive_job_configuration function."""

    def test_empty_vtt_files(self) -> None:
        """Test configuration with empty VTT files list."""
        console = Console()
        result = interactive_job_configuration([], Path("/test"), console=console)
        assert result == []

    @patch("vtt2minutes.batch._prompt_for_title")
    @patch("vtt2minutes.batch._prompt_for_output_path")
    @patch("vtt2minutes.batch._prompt_for_processing_confirmation")
    def test_single_file_configuration(
        self, mock_process, mock_output, mock_title
    ) -> None:
        """Test configuring single VTT file."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file = base_dir / "meeting.vtt"
            vtt_file.touch()

            mock_title.return_value = "Test Meeting"
            mock_output.return_value = base_dir / "test_output.md"
            mock_process.return_value = True

            console = Console()
            result = interactive_job_configuration(
                [vtt_file], base_dir, console=console
            )

            assert len(result) == 1
            assert result[0].vtt_file == vtt_file
            assert result[0].title == "Test Meeting"
            assert result[0].output_file == base_dir / "test_output.md"
            assert result[0].enabled is True

    @patch("vtt2minutes.batch._prompt_for_title")
    @patch("vtt2minutes.batch._prompt_for_output_path")
    @patch("vtt2minutes.batch._prompt_for_processing_confirmation")
    def test_skip_file_configuration(
        self, mock_process, mock_output, mock_title
    ) -> None:
        """Test skipping file in configuration."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file = base_dir / "meeting.vtt"
            vtt_file.touch()

            mock_title.return_value = "Test Meeting"
            mock_output.return_value = base_dir / "test_output.md"
            mock_process.return_value = False  # Skip this file

            console = Console()
            result = interactive_job_configuration(
                [vtt_file], base_dir, console=console
            )

            assert len(result) == 0


class TestReviewAndConfirmJobs:
    """Tests for review_and_confirm_jobs function."""

    def test_empty_jobs(self) -> None:
        """Test review with empty jobs list."""
        console = Console()
        proceed, jobs = review_and_confirm_jobs([], Path("/test"), console=console)
        assert proceed is False
        assert jobs == []

    @patch("vtt2minutes.batch._get_user_choice")
    def test_proceed_with_jobs(self, mock_choice) -> None:
        """Test proceeding with jobs without conflicts."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file = base_dir / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            mock_choice.return_value = "y"

            console = Console()
            proceed, jobs = review_and_confirm_jobs([job], base_dir, console=console)

            assert proceed is True
            assert len(jobs) == 1

    @patch("vtt2minutes.batch._get_user_choice")
    def test_cancel_jobs(self, mock_choice) -> None:
        """Test cancelling job review."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file = base_dir / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            mock_choice.return_value = "n"

            console = Console()
            proceed, jobs = review_and_confirm_jobs([job], base_dir, console=console)

            assert proceed is False
            assert jobs == []


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_initialization(self) -> None:
        """Test processor initialization."""
        config = ProcessingConfig()
        processor = BatchProcessor(config)

        assert processor.config == config
        assert processor.console is not None

    def test_process_empty_jobs(self) -> None:
        """Test processing empty jobs list."""
        config = ProcessingConfig()
        processor = BatchProcessor(config)

        results = processor.process_batch([])
        assert results == []

    def test_process_disabled_jobs(self) -> None:
        """Test processing with all jobs disabled."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file = base_dir / "meeting.vtt"
            vtt_file.touch()

            job = BatchJob.from_vtt_file(vtt_file)
            job.enabled = False

            config = ProcessingConfig()
            processor = BatchProcessor(config)

            results = processor.process_batch([job])
            assert results == []

    @patch("vtt2minutes.batch.VTTFileProcessor")
    def test_process_successful_job(self, mock_processor_class) -> None:
        """Test processing successful job."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file = base_dir / "meeting.vtt"
            output_file = base_dir / "output.md"
            vtt_file.touch()

            job = BatchJob(
                vtt_file=vtt_file,
                title="Test Meeting",
                output_file=output_file,
                enabled=True,
            )

            # Mock successful processing
            mock_processor = Mock()
            mock_result = ProcessingResult(
                success=True, input_file=vtt_file, output_file=output_file
            )
            mock_processor.process_file.return_value = mock_result
            mock_processor_class.return_value = mock_processor

            config = ProcessingConfig()
            processor = BatchProcessor(config)

            results = processor.process_batch([job])

            assert len(results) == 1
            assert results[0].success is True
            assert results[0].input_file == vtt_file

    @patch("vtt2minutes.batch.VTTFileProcessor")
    @patch("vtt2minutes.batch.Confirm.ask")
    def test_process_failed_job_continue(
        self, mock_confirm, mock_processor_class
    ) -> None:
        """Test processing failed job with continuation."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            vtt_file1 = base_dir / "meeting1.vtt"
            vtt_file2 = base_dir / "meeting2.vtt"
            vtt_file1.touch()
            vtt_file2.touch()

            job1 = BatchJob.from_vtt_file(vtt_file1)
            job2 = BatchJob.from_vtt_file(vtt_file2)

            # Mock first job failing, second succeeding
            mock_processor = Mock()
            mock_result1 = ProcessingResult(
                success=False, input_file=vtt_file1, error="Test error"
            )
            mock_result2 = ProcessingResult(
                success=True, input_file=vtt_file2, output_file=base_dir / "meeting2.md"
            )
            mock_processor.process_file.side_effect = [mock_result1, mock_result2]
            mock_processor_class.return_value = mock_processor

            # User chooses to continue after error
            mock_confirm.return_value = True

            config = ProcessingConfig()
            processor = BatchProcessor(config)

            results = processor.process_batch([job1, job2])

            assert len(results) == 2
            assert results[0].success is False
            assert results[1].success is True

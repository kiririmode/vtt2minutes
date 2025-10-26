"""Tests for processor module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from vtt2minutes.processor import ProcessingConfig, ProcessingResult, VTTFileProcessor


class TestProcessingConfig:
    """Tests for ProcessingConfig class."""

    def test_default_config(self) -> None:
        """Test creating config with default values."""
        config = ProcessingConfig()

        assert config.no_preprocessing is False
        assert config.min_duration == 0.5
        assert config.merge_threshold == 2.0
        assert config.duplicate_threshold == 0.8
        assert config.filter_words_file is None
        assert config.replacement_rules_file is None
        assert config.bedrock_model is None
        assert config.bedrock_inference_profile_id is None
        assert config.bedrock_region == "ap-northeast-1"
        assert config.prompt_template is None
        assert config.overwrite is False
        assert config.verbose is False
        assert config.stats is False
        assert config.keep_vtt is False

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = ProcessingConfig(
            no_preprocessing=True,
            min_duration=1.0,
            bedrock_model="anthropic.claude-3-sonnet-20241022-v2:0",
            bedrock_region="us-east-1",
            overwrite=True,
            verbose=True,
            stats=True,
        )

        assert config.no_preprocessing is True
        assert config.min_duration == 1.0
        assert config.bedrock_model == "anthropic.claude-3-sonnet-20241022-v2:0"
        assert config.bedrock_region == "us-east-1"
        assert config.overwrite is True
        assert config.verbose is True
        assert config.stats is True


class TestProcessingResult:
    """Tests for ProcessingResult class."""

    def test_successful_result(self) -> None:
        """Test creating successful processing result."""
        input_file = Path("input.vtt")
        output_file = Path("output.md")
        intermediate_file = Path("intermediate.md")

        result = ProcessingResult(
            success=True,
            input_file=input_file,
            output_file=output_file,
            intermediate_file=intermediate_file,
        )

        assert result.success is True
        assert result.input_file == input_file
        assert result.output_file == output_file
        assert result.intermediate_file == intermediate_file
        assert result.error is None
        assert result.statistics is None

    def test_failed_result(self) -> None:
        """Test creating failed processing result."""
        input_file = Path("input.vtt")
        error_message = "Processing failed"

        result = ProcessingResult(
            success=False, input_file=input_file, error=error_message
        )

        assert result.success is False
        assert result.input_file == input_file
        assert result.output_file is None
        assert result.intermediate_file is None
        assert result.error == error_message
        assert result.statistics is None


class TestVTTFileProcessor:
    """Tests for VTTFileProcessor class."""

    def test_initialization(self) -> None:
        """Test processor initialization."""
        config = ProcessingConfig()
        processor = VTTFileProcessor(config)

        assert processor.config == config
        assert processor.console is not None

    @patch("vtt2minutes.processor.VTTParser")
    @patch("vtt2minutes.processor.TextPreprocessor")
    def test_initialize_components(
        self, mock_preprocessor_class, mock_parser_class
    ) -> None:
        """Test component initialization."""
        config = ProcessingConfig(
            min_duration=1.0, merge_threshold=3.0, duplicate_threshold=0.9
        )
        processor = VTTFileProcessor(config)

        mock_parser = Mock()
        mock_preprocessor = Mock()
        mock_parser_class.return_value = mock_parser
        mock_preprocessor_class.return_value = mock_preprocessor

        parser, preprocessor = processor._initialize_components()

        assert parser == mock_parser
        assert preprocessor == mock_preprocessor
        mock_parser_class.assert_called_once()
        mock_preprocessor_class.assert_called_once()

    @patch("vtt2minutes.processor.VTTParser")
    def test_parse_vtt_file_success(self, mock_parser_class) -> None:
        """Test successful VTT file parsing."""
        config = ProcessingConfig(verbose=True)
        processor = VTTFileProcessor(config)

        mock_parser = Mock()
        mock_cues = [Mock(), Mock()]
        mock_parser.parse_file.return_value = mock_cues
        mock_parser.get_speakers.return_value = ["Alice", "Bob"]
        mock_parser.get_duration.return_value = 120.0

        mock_progress = Mock()
        mock_task = Mock()
        mock_progress.add_task.return_value = mock_task

        input_file = Path("test.vtt")
        result = processor._parse_vtt_file(mock_parser, input_file, mock_progress)

        assert result == mock_cues
        mock_parser.parse_file.assert_called_once_with(input_file)
        mock_progress.add_task.assert_called_once()
        mock_progress.update.assert_called_once()

    @patch("vtt2minutes.processor.VTTParser")
    def test_parse_vtt_file_error(self, mock_parser_class) -> None:
        """Test VTT file parsing error."""
        config = ProcessingConfig()
        processor = VTTFileProcessor(config)

        mock_parser = Mock()
        mock_parser.parse_file.side_effect = Exception("Parse error")

        mock_progress = Mock()
        mock_progress.add_task.return_value = Mock()

        input_file = Path("test.vtt")

        with pytest.raises(RuntimeError, match="VTTファイルの解析に失敗しました"):
            processor._parse_vtt_file(mock_parser, input_file, mock_progress)

    @patch("vtt2minutes.processor.TextPreprocessor")
    def test_preprocess_cues_disabled(self, mock_preprocessor_class) -> None:
        """Test preprocessing when disabled."""
        config = ProcessingConfig(no_preprocessing=True)
        processor = VTTFileProcessor(config)

        mock_cues = [Mock(), Mock()]
        mock_progress = Mock()

        result = processor._preprocess_cues(Mock(), mock_cues, mock_progress)

        assert result == mock_cues

    @patch("vtt2minutes.processor.TextPreprocessor")
    def test_preprocess_cues_enabled(self, mock_preprocessor_class) -> None:
        """Test preprocessing when enabled."""
        config = ProcessingConfig(no_preprocessing=False, verbose=True)
        processor = VTTFileProcessor(config)

        mock_cues = [Mock(), Mock(), Mock()]
        mock_processed_cues = [Mock(), Mock()]
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_cues.return_value = mock_processed_cues

        mock_progress = Mock()
        mock_task = Mock()
        mock_progress.add_task.return_value = mock_task

        result = processor._preprocess_cues(mock_preprocessor, mock_cues, mock_progress)

        assert result == mock_processed_cues
        mock_preprocessor.preprocess_cues.assert_called_once_with(mock_cues)

    @patch("vtt2minutes.processor.IntermediateTranscriptWriter")
    def test_save_intermediate_file_success(self, mock_writer_class) -> None:
        """Test successful intermediate file saving."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(overwrite=True, verbose=True)
            processor = VTTFileProcessor(config)

            mock_cues = [Mock()]
            output = Path(temp_dir) / "output.md"
            title = "Test Meeting"

            mock_writer = Mock()
            mock_stats = {
                "speakers": ["Alice", "Bob"],
                "duration": 120.0,
                "word_count": 1000,
            }
            mock_writer.get_statistics.return_value = mock_stats
            mock_writer.format_duration.return_value = "2:00"
            mock_writer_class.return_value = mock_writer

            mock_progress = Mock()
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task

            result = processor._save_intermediate_file(
                mock_cues, output, None, title, mock_progress
            )

            expected_path = output.with_suffix(".preprocessed.md")
            assert result == expected_path
            mock_writer.write_markdown.assert_called_once()

    @patch("vtt2minutes.processor.IntermediateTranscriptWriter")
    def test_save_intermediate_file_exists_no_overwrite(
        self, mock_writer_class
    ) -> None:
        """Test intermediate file saving when file exists and overwrite disabled."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(overwrite=False)
            processor = VTTFileProcessor(config)

            output = Path(temp_dir) / "output.md"
            intermediate_path = output.with_suffix(".preprocessed.md")
            intermediate_path.touch()  # Create existing file

            mock_progress = Mock()
            mock_progress.add_task.return_value = Mock()

            with pytest.raises(RuntimeError, match="中間ファイルの保存に失敗しました"):
                processor._save_intermediate_file(
                    [Mock()], output, None, "Test", mock_progress
                )

    @patch("vtt2minutes.processor.BedrockMeetingMinutesGenerator")
    def test_generate_chat_prompt_success(self, mock_generator_class) -> None:
        """Test successful chat prompt generation."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(
                bedrock_region="us-east-1",
                prompt_template=None,
                overwrite=True,
                verbose=True,
            )
            processor = VTTFileProcessor(config)

            intermediate_path = Path(temp_dir) / "intermediate.md"
            intermediate_path.write_text("Test content")
            chat_prompt_file = Path(temp_dir) / "prompt.txt"
            title = "Test Meeting"

            mock_generator = Mock()
            mock_generator.create_chat_prompt.return_value = "Generated prompt"
            mock_generator_class.return_value = mock_generator

            mock_progress = Mock()
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task

            processor._generate_chat_prompt(
                intermediate_path, chat_prompt_file, title, mock_progress
            )

            assert chat_prompt_file.exists()
            assert chat_prompt_file.read_text() == "Generated prompt"
            mock_generator.create_chat_prompt.assert_called_once_with(
                "Test content", "Test Meeting"
            )

    @patch("vtt2minutes.processor.BedrockMeetingMinutesGenerator")
    def test_generate_bedrock_minutes_success(self, mock_generator_class) -> None:
        """Test successful Bedrock minutes generation."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(
                bedrock_model="anthropic.claude-3-sonnet-20241022-v2:0",
                bedrock_region="us-east-1",
                verbose=True,
            )
            processor = VTTFileProcessor(config)

            intermediate_path = Path(temp_dir) / "intermediate.md"
            intermediate_path.write_text("Test content")
            title = "Test Meeting"

            mock_generator = Mock()
            mock_generator.generate_minutes_from_markdown.return_value = (
                "Generated minutes"
            )
            mock_generator_class.return_value = mock_generator

            mock_progress = Mock()
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task

            result = processor._generate_bedrock_minutes(
                intermediate_path, title, mock_progress
            )

            assert result == "Generated minutes"
            mock_generator.generate_minutes_from_markdown.assert_called_once_with(
                "Test content", title="Test Meeting"
            )

    def test_validate_bedrock_config_both_specified(self) -> None:
        """Test validation error when both model and profile specified."""
        config = ProcessingConfig(
            bedrock_model="model", bedrock_inference_profile_id="profile"
        )
        processor = VTTFileProcessor(config)

        with pytest.raises(ValueError, match="両方を指定することは"):
            processor._validate_bedrock_config()

    def test_validate_bedrock_config_none_specified(self) -> None:
        """Test validation error when neither model nor profile specified."""
        config = ProcessingConfig()
        processor = VTTFileProcessor(config)

        with pytest.raises(ValueError, match="どちらか一方を"):
            processor._validate_bedrock_config()

    def test_validate_bedrock_config_model_only(self) -> None:
        """Test validation success with model only."""
        config = ProcessingConfig(bedrock_model="model")
        processor = VTTFileProcessor(config)

        # Should not raise exception
        processor._validate_bedrock_config()

    def test_validate_bedrock_config_profile_only(self) -> None:
        """Test validation success with profile only."""
        config = ProcessingConfig(bedrock_inference_profile_id="profile")
        processor = VTTFileProcessor(config)

        # Should not raise exception
        processor._validate_bedrock_config()

    def test_save_output_file_success(self) -> None:
        """Test successful output file saving."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(overwrite=True)
            processor = VTTFileProcessor(config)

            output = Path(temp_dir) / "output.md"
            content = "Test content"

            mock_progress = Mock()
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task

            processor._save_output_file(output, content, mock_progress)

            assert output.exists()
            assert output.read_text() == content

    def test_save_output_file_exists_no_overwrite(self) -> None:
        """Test output file saving when file exists and overwrite disabled."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(overwrite=False)
            processor = VTTFileProcessor(config)

            output = Path(temp_dir) / "output.md"
            output.touch()  # Create existing file

            mock_progress = Mock()
            mock_progress.add_task.return_value = Mock()

            with pytest.raises(RuntimeError, match="ファイルの保存に失敗しました"):
                processor._save_output_file(output, "content", mock_progress)

    @patch("vtt2minutes.processor.TextPreprocessor")
    def test_collect_statistics_no_preprocessing(self, mock_preprocessor_class) -> None:
        """Test statistics collection when preprocessing disabled."""
        config = ProcessingConfig(no_preprocessing=True, stats=True)
        processor = VTTFileProcessor(config)

        result = processor._collect_statistics([], [], Mock())

        assert result == {}

    @patch("vtt2minutes.processor.TextPreprocessor")
    def test_collect_statistics_enabled(self, mock_preprocessor_class) -> None:
        """Test statistics collection when enabled."""
        config = ProcessingConfig(no_preprocessing=False, stats=True)
        processor = VTTFileProcessor(config)

        original_cues = [Mock(), Mock()]
        processed_cues = [Mock()]
        mock_preprocessor = Mock()
        expected_stats = {"original_count": 2, "processed_count": 1}
        mock_preprocessor.get_statistics.return_value = expected_stats

        result = processor._collect_statistics(
            original_cues, processed_cues, mock_preprocessor
        )

        assert result == expected_stats
        mock_preprocessor.get_statistics.assert_called_once_with(
            original_cues, processed_cues
        )

    @patch("vtt2minutes.processor.VTTParser")
    @patch("vtt2minutes.processor.TextPreprocessor")
    @patch("vtt2minutes.processor.IntermediateTranscriptWriter")
    @patch("vtt2minutes.processor.BedrockMeetingMinutesGenerator")
    def test_process_file_with_chat_prompt(
        self,
        mock_generator_class,
        mock_writer_class,
        mock_preprocessor_class,
        mock_parser_class,
    ) -> None:
        """Test complete file processing with chat prompt generation."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(overwrite=True)
            processor = VTTFileProcessor(config)

            input_file = Path(temp_dir) / "input.vtt"
            output_file = Path(temp_dir) / "output.md"
            chat_prompt_file = Path(temp_dir) / "prompt.txt"
            title = "Test Meeting"

            # Create intermediate file that will be expected
            intermediate_file = output_file.with_suffix(".preprocessed.md")
            intermediate_file.write_text("Test intermediate content")

            # Setup mocks
            mock_parser = Mock()
            mock_cues = [Mock()]
            mock_parser.parse_file.return_value = mock_cues
            mock_parser_class.return_value = mock_parser

            mock_preprocessor = Mock()
            mock_preprocessor.preprocess_cues.return_value = mock_cues
            mock_preprocessor_class.return_value = mock_preprocessor

            mock_writer = Mock()
            mock_writer.get_statistics.return_value = {
                "speakers": ["Alice"],
                "duration": 60.0,
                "word_count": 500,
            }
            mock_writer.format_duration.return_value = "1:00"
            mock_writer_class.return_value = mock_writer

            mock_generator = Mock()
            mock_generator.create_chat_prompt.return_value = "Generated prompt"
            mock_generator_class.return_value = mock_generator

            result = processor.process_file(
                input_file, output_file, title, chat_prompt_file=chat_prompt_file
            )

            assert result.success is True
            assert result.input_file == input_file
            assert result.output_file == chat_prompt_file
            assert result.intermediate_file is not None

    @patch("vtt2minutes.processor.VTTParser")
    def test_process_file_error_handling(self, mock_parser_class) -> None:
        """Test process file error handling."""
        config = ProcessingConfig()
        processor = VTTFileProcessor(config)

        input_file = Path("input.vtt")
        output_file = Path("output.md")

        mock_parser = Mock()
        mock_parser.parse_file.side_effect = Exception("Parse error")
        mock_parser_class.return_value = mock_parser

        result = processor.process_file(input_file, output_file)

        assert result.success is False
        assert result.input_file == input_file
        assert result.error is not None
        assert "Parse error" in result.error

    def test_delete_vtt_file_success(self) -> None:
        """Test successful VTT file deletion."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(keep_vtt=False, verbose=True)
            processor = VTTFileProcessor(config)

            vtt_file = Path(temp_dir) / "test.vtt"
            vtt_file.write_text("WEBVTT")

            processor._delete_vtt_file(vtt_file)

            assert not vtt_file.exists()

    def test_delete_vtt_file_failure(self) -> None:
        """Test VTT file deletion failure handling."""
        config = ProcessingConfig(keep_vtt=False, verbose=False)
        processor = VTTFileProcessor(config)

        # Try to delete non-existent file
        vtt_file = Path("/nonexistent/test.vtt")

        # Should not raise exception, only log warning
        processor._delete_vtt_file(vtt_file)

    @patch("vtt2minutes.processor.VTTParser")
    @patch("vtt2minutes.processor.TextPreprocessor")
    @patch("vtt2minutes.processor.IntermediateTranscriptWriter")
    @patch("vtt2minutes.processor.BedrockMeetingMinutesGenerator")
    def test_process_file_deletes_vtt_on_success(
        self,
        mock_generator_class,
        mock_writer_class,
        mock_preprocessor_class,
        mock_parser_class,
    ) -> None:
        """Test VTT file is deleted after successful processing."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(
                bedrock_model="anthropic.claude-3-sonnet-20241022-v2:0",
                overwrite=True,
                keep_vtt=False,
            )
            processor = VTTFileProcessor(config)

            input_file = Path(temp_dir) / "input.vtt"
            input_file.write_text("WEBVTT")
            output_file = Path(temp_dir) / "output.md"
            title = "Test Meeting"

            intermediate_file = output_file.with_suffix(".preprocessed.md")
            intermediate_file.write_text("Test intermediate content")

            # Setup mocks
            mock_parser = Mock()
            mock_cues = [Mock()]
            mock_parser.parse_file.return_value = mock_cues
            mock_parser_class.return_value = mock_parser

            mock_preprocessor = Mock()
            mock_preprocessor.preprocess_cues.return_value = mock_cues
            mock_preprocessor_class.return_value = mock_preprocessor

            mock_writer = Mock()
            mock_writer.get_statistics.return_value = {
                "speakers": ["Alice"],
                "duration": 60.0,
                "word_count": 500,
            }
            mock_writer.format_duration.return_value = "1:00"
            mock_writer_class.return_value = mock_writer

            mock_generator = Mock()
            mock_generator.generate_minutes_from_markdown.return_value = (
                "Generated minutes"
            )
            mock_generator_class.return_value = mock_generator

            result = processor.process_file(input_file, output_file, title)

            assert result.success is True
            assert not input_file.exists()

    @patch("vtt2minutes.processor.VTTParser")
    @patch("vtt2minutes.processor.TextPreprocessor")
    @patch("vtt2minutes.processor.IntermediateTranscriptWriter")
    @patch("vtt2minutes.processor.BedrockMeetingMinutesGenerator")
    def test_process_file_keeps_vtt_when_requested(
        self,
        mock_generator_class,
        mock_writer_class,
        mock_preprocessor_class,
        mock_parser_class,
    ) -> None:
        """Test VTT file is kept when keep_vtt is True."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(
                bedrock_model="anthropic.claude-3-sonnet-20241022-v2:0",
                overwrite=True,
                keep_vtt=True,
            )
            processor = VTTFileProcessor(config)

            input_file = Path(temp_dir) / "input.vtt"
            input_file.write_text("WEBVTT")
            output_file = Path(temp_dir) / "output.md"
            title = "Test Meeting"

            intermediate_file = output_file.with_suffix(".preprocessed.md")
            intermediate_file.write_text("Test intermediate content")

            # Setup mocks
            mock_parser = Mock()
            mock_cues = [Mock()]
            mock_parser.parse_file.return_value = mock_cues
            mock_parser_class.return_value = mock_parser

            mock_preprocessor = Mock()
            mock_preprocessor.preprocess_cues.return_value = mock_cues
            mock_preprocessor_class.return_value = mock_preprocessor

            mock_writer = Mock()
            mock_writer.get_statistics.return_value = {
                "speakers": ["Alice"],
                "duration": 60.0,
                "word_count": 500,
            }
            mock_writer.format_duration.return_value = "1:00"
            mock_writer_class.return_value = mock_writer

            mock_generator = Mock()
            mock_generator.generate_minutes_from_markdown.return_value = (
                "Generated minutes"
            )
            mock_generator_class.return_value = mock_generator

            result = processor.process_file(input_file, output_file, title)

            assert result.success is True
            assert input_file.exists()

    @patch("vtt2minutes.processor.VTTParser")
    def test_process_file_keeps_vtt_on_failure(self, mock_parser_class) -> None:
        """Test VTT file is kept when processing fails."""
        with TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(keep_vtt=False)
            processor = VTTFileProcessor(config)

            input_file = Path(temp_dir) / "input.vtt"
            input_file.write_text("WEBVTT")
            output_file = Path(temp_dir) / "output.md"

            mock_parser = Mock()
            mock_parser.parse_file.side_effect = Exception("Parse error")
            mock_parser_class.return_value = mock_parser

            result = processor.process_file(input_file, output_file)

            assert result.success is False
            assert input_file.exists()

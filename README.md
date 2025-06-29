# VTT2Minutes

A Python tool that automatically generates high-quality meeting minutes from Microsoft Teams transcript files (VTT).

## Features

- **VTT File Parsing**: Analyze Microsoft Teams WebVTT format transcripts
- **Advanced Preprocessing**: Improve transcription quality through filler word removal, noise reduction, and duplicate elimination
- **Japanese & English Support**: Handle filler words and punctuation in both languages
- **Automatic Meeting Minutes Generation**: Output structured meeting minutes in Markdown format
- **Speaker Identification**: Identify and properly categorize each speaker's contributions
- **Action Item Extraction**: Automatically detect action items and decisions

## Installation

### Requirements

- Python 3.12 or higher
- uv (recommended package manager)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd vtt2minutes

# Install dependencies
uv sync

# Install development dependencies (for developers)
uv sync --extra dev
```

## Usage

### Basic Usage

```bash
# Generate meeting minutes from VTT file
uv run python -m vtt2minutes meeting.vtt

# Specify output file name
uv run python -m vtt2minutes meeting.vtt -o minutes.md

# Specify meeting title
uv run python -m vtt2minutes meeting.vtt -t "Project Planning Meeting"
```

### Advanced Options

```bash
# Show detailed output and statistics
uv run python -m vtt2minutes meeting.vtt --verbose --stats

# Skip preprocessing
uv run python -m vtt2minutes meeting.vtt --no-preprocessing

# Customize preprocessing settings
uv run python -m vtt2minutes meeting.vtt \
  --min-duration 1.0 \
  --merge-threshold 3.0 \
  --duplicate-threshold 0.9
```

### File Information

```bash
# Display VTT file information without processing
uv run python -m vtt2minutes info meeting.vtt
```

## Output Example

Example of generated meeting minutes:

```markdown
# Project Planning Meeting

**Date:** June 29, 2025
**Duration:** 00:15:23
**Participants:** Tanaka, Sato, Yamada

## Summary
This meeting included 3 participants with a total of 156 recorded words. The main discussion focused on new feature specification review and schedule coordination.

## Key Points
- Discussed UI design for new features
- Need to complete specifications by next week
- Test environment preparation required

## Decisions
1. Decided to set new feature release date for end of next month
   - Decided at: 00:08:15

## Action Items
1. Complete specification document creation
   - Assignee: Tanaka
   - Due date: Next week
   - Mentioned at: 00:05:30

## Detailed Minutes
### Tanaka's Comments
**Time:** 00:01:00 - 00:02:30
**Speaker:** Tanaka

I would like to discuss the new features today. First, let me share the current progress...
```

## Configuration Options

### Preprocessing Settings

- `--min-duration`: Minimum time to recognize as valid speech (seconds)
- `--merge-threshold`: Time gap threshold for merging consecutive statements (seconds)
- `--duplicate-threshold`: Similarity threshold for duplicate detection (0.0-1.0)
- `--no-preprocessing`: Skip preprocessing entirely

### Output Settings

- `--output, -o`: Output file path
- `--title, -t`: Meeting title
- `--verbose, -v`: Show detailed processing status
- `--stats`: Display preprocessing statistics

## Supported Filler Words

### Japanese
- えー, あー, うー, そのー, なんか, まあ, ちょっと
- えっと, あのー, そうですね, はい, ええ, うん

### English  
- um, uh, like, you know, so, well, okay, right
- actually, basically, literally, honestly

### Transcription Artifacts
- [音声が途切れました], [雑音], [不明瞭], [咳], [笑い]
- [audio interrupted], [noise], [unclear], [cough], [laughter]

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/vtt2minutes

# Run specific test file only
uv run pytest tests/test_parser.py
```

### Code Quality Checks

```bash
# Code formatting
uv run ruff format .

# Run linting
uv run ruff check .

# Type checking
uv run pyright
```

### Pre-commit Checks

```bash
# Run formatting and linting together
uv run ruff format . && uv run ruff check . --fix
```

## Architecture

```
src/vtt2minutes/
├── __init__.py          # Package entry point
├── parser.py            # VTT file parsing
├── preprocessor.py      # Text preprocessing
├── summarizer.py        # Meeting minutes generation
└── cli.py              # Command-line interface
```

### Main Classes

- **VTTParser**: WebVTT file parsing and VTTCue object generation
- **TextPreprocessor**: Filler word removal, duplicate elimination, text cleaning
- **MeetingSummarizer**: Meeting minutes structure generation and Markdown output

## Technical Specifications

- **Python**: 3.12 or higher
- **Dependencies**: Click (CLI), Rich (UI), standard library only otherwise
- **Input Format**: WebVTT (.vtt)
- **Output Format**: Markdown (.md)
- **Character Encoding**: UTF-8

## License

MIT License

## Contributing

Pull requests and issue reports are welcome. To contribute to development:

1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Run tests to ensure everything passes
5. Create a pull request

For detailed development guidelines, see `CLAUDE.md`.

## Use Cases

- **Meeting Documentation**: Convert Teams meeting transcripts into professional minutes
- **Content Analysis**: Extract key decisions and action items from long discussions
- **Multilingual Support**: Handle Japanese and English meetings seamlessly
- **Quality Improvement**: Clean up automatic transcription artifacts for better readability
- **Time Saving**: Automate the manual process of creating structured meeting notes

## Limitations

- Currently supports Microsoft Teams VTT format specifically
- Automatic action item and decision detection may require manual review for accuracy
- Best results with clear audio and distinct speakers
- Some context-dependent filler word detection may vary by meeting style
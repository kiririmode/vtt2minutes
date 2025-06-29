# VTT2Minutes

A Python tool that automatically generates AI-powered meeting minutes from Microsoft Teams transcript files (VTT) using Amazon Bedrock.

## Features

- **VTT File Parsing**: Analyze Microsoft Teams WebVTT format transcripts
- **Advanced Preprocessing**: Improve transcription quality through filler word removal, noise reduction, and duplicate elimination
- **Japanese & English Support**: Handle filler words and punctuation in both languages
- **AI-Powered Meeting Minutes**: Generate intelligent, context-aware meeting minutes using Amazon Bedrock
- **Speaker Identification**: Identify and properly categorize each speaker's contributions
- **Intermediate File Output**: Export preprocessed transcripts in structured Markdown format

## Installation

### Requirements

- Python 3.12 or higher
- uv (recommended package manager)
- AWS Account with Amazon Bedrock access
- AWS credentials configured

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd vtt2minutes

# Install dependencies
uv sync

# Install development dependencies (for developers)
uv sync --extra dev

# Configure AWS credentials (required)
# Copy the sample environment file and edit it with your credentials
cp .env.sample .env
# Or use: export AWS_ACCESS_KEY_ID=your-access-key
# Or use: aws configure
```

## Usage

### Basic Usage

```bash
# Generate AI-powered meeting minutes from VTT file
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

# Use custom filler words file
uv run python -m vtt2minutes meeting.vtt --filter-words-file my_filter_words.txt

# Specify Bedrock model and region
uv run python -m vtt2minutes meeting.vtt \
  --bedrock-model anthropic.claude-3-sonnet-20240229-v1:0 \
  --bedrock-region us-west-2

# Save intermediate preprocessed file
uv run python -m vtt2minutes meeting.vtt --intermediate-file preprocessed.md
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
- `--filter-words-file`: Path to custom filler words file
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

## Custom Filler Words File

You can create a custom filler words file to override the default filler words. The file format is simple:

```txt
# My custom filler words
# Lines starting with # are comments and will be ignored
# Empty lines are also ignored

# Custom Japanese filler words
えっと
そのー
まぁ

# Custom English filler words
basically
obviously
definitely

# Custom transcription artifacts
[microphone feedback]
[phone ringing]
```

### Usage Example

```bash
# Create your custom filter words file
echo "basically" > my_filters.txt
echo "obviously" >> my_filters.txt
echo "えっと" >> my_filters.txt

# Use it with vtt2minutes
uv run python -m vtt2minutes meeting.vtt --filter-words-file my_filters.txt
```

**Note**: When using a custom filter words file, it completely replaces the default filler words. If you want to keep some default words, include them in your custom file.

## Amazon Bedrock Integration

VTT2Minutes supports AI-powered meeting minutes generation using Amazon Bedrock, providing more sophisticated and context-aware output compared to traditional keyword-based summarization.

### Prerequisites

1. **AWS Account**: You need an active AWS account with Bedrock access
2. **Bedrock Model Access**: Request access to Claude models in your AWS region
3. **AWS Credentials**: Configure your AWS credentials using environment variables

### AWS Credentials Setup

#### Option 1: Using Environment Variables (.env file) - Recommended

```bash
# Copy the sample environment file
cp .env.sample .env

# Edit .env file with your actual AWS credentials
# The file contains detailed comments explaining each variable
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_SESSION_TOKEN=your-session-token  # Optional, for temporary credentials
```

#### Option 2: Export Environment Variables

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_SESSION_TOKEN=your-session-token  # Optional, for temporary credentials
```

#### Option 3: Using AWS CLI Profile

```bash
# Or using AWS CLI profile
aws configure set aws_access_key_id your-access-key-id
aws configure set aws_secret_access_key your-secret-access-key
aws configure set region ap-northeast-1
```

**Note**: The `.env.sample` file contains comprehensive documentation about all configuration options, security best practices, and alternative authentication methods.

### Supported Models

- **Claude 3 Haiku** (default): `anthropic.claude-3-haiku-20240307-v1:0`
- **Claude 3 Sonnet**: `anthropic.claude-3-sonnet-20240229-v1:0`
- **Claude 3 Opus**: `anthropic.claude-3-opus-20240229-v1:0`

### Bedrock Options

- `--bedrock-model`: Specify the Bedrock model ID to use (mutually exclusive with --bedrock-inference-profile-id)
- `--bedrock-inference-profile-id`: Specify the Bedrock inference profile ID to use (mutually exclusive with --bedrock-model)
- `--bedrock-region`: AWS region for Bedrock (default: ap-northeast-1)
- `--intermediate-file`: Path to save intermediate preprocessed file

### Example Usage

```bash
# Basic usage (uses Claude 3 Haiku by default)
uv run python -m vtt2minutes meeting.vtt

# With custom model and region
uv run python -m vtt2minutes meeting.vtt \
  --bedrock-model anthropic.claude-3-sonnet-20240229-v1:0 \
  --bedrock-region us-west-2

# Using APAC inference profile ID (default region: ap-northeast-1)
uv run python -m vtt2minutes meeting.vtt \
  --bedrock-inference-profile-id apac.anthropic.claude-sonnet-4-20250514-v1:0

# Save intermediate file for inspection
uv run python -m vtt2minutes meeting.vtt \
  --intermediate-file meeting_preprocessed.md \
  --output ai_minutes.md
```

### How It Works

1. **Preprocessing**: VTT file is processed to remove filler words and clean up transcription
2. **Intermediate File**: Preprocessed content is saved in structured Markdown format
3. **AI Generation**: Bedrock model analyzes the intermediate file and generates comprehensive meeting minutes
4. **Output**: AI-generated minutes are saved in final output file

### Benefits of AI-Powered Minutes

- **Context Awareness**: Better understanding of meeting flow and relationships between topics
- **Intelligent Summarization**: More accurate extraction of key points and decisions
- **Natural Language**: Output reads more naturally compared to keyword-based extraction
- **Adaptive Processing**: Handles various meeting styles and formats effectively

### Costs and Considerations

- **AWS Charges**: Using Bedrock incurs AWS charges based on model usage
- **Processing Time**: AI generation takes longer than traditional processing
- **Network Required**: Requires internet connection to AWS Bedrock service
- **Token Limits**: Large meetings may need to be processed in chunks

### Troubleshooting

Common issues and solutions:

```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify Bedrock access in your region
aws bedrock list-foundation-models --region us-east-1

# Test with verbose output for debugging
uv run python -m vtt2minutes meeting.vtt --verbose
```

**Error Messages:**
- `Missing environment variables`: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
- `Bedrock is not available in region`: Use a supported region like us-east-1
- `Invalid credentials`: Check your AWS access keys and permissions

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
├── intermediate.py      # Intermediate file output
├── bedrock.py           # Amazon Bedrock integration
└── cli.py              # Command-line interface
```

### Main Classes

- **VTTParser**: WebVTT file parsing and VTTCue object generation
- **TextPreprocessor**: Filler word removal, duplicate elimination, text cleaning
- **IntermediateTranscriptWriter**: Preprocessed transcript output in Markdown format
- **BedrockMeetingMinutesGenerator**: AI-powered meeting minutes using Amazon Bedrock

## Technical Specifications

- **Python**: 3.12 or higher
- **Dependencies**: Click (CLI), Rich (UI), boto3 (AWS SDK), standard library otherwise
- **Input Format**: WebVTT (.vtt)
- **Output Format**: Markdown (.md)
- **Intermediate Format**: Structured Markdown (.md)
- **Character Encoding**: UTF-8
- **Cloud Integration**: Amazon Bedrock (required)

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
- **AI-Enhanced Processing**: Generate intelligent, context-aware meeting minutes using Amazon Bedrock
- **Enterprise Integration**: Scale meeting documentation with cloud-based AI services

## Limitations

- Currently supports Microsoft Teams VTT format specifically
- Requires AWS account and Bedrock access (incurs AWS charges)
- Best results with clear audio and distinct speakers
- Some context-dependent filler word detection may vary by meeting style
- Internet connection required for AI processing

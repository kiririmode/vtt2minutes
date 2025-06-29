"""VTT2Minutes - Convert Teams VTT transcripts to AI-powered meeting minutes."""

__version__ = "0.1.0"
__author__ = "VTT2Minutes Contributors"
__description__ = (
    "Convert Microsoft Teams VTT transcripts to AI-powered meeting minutes "
    "using Amazon Bedrock"
)

from .bedrock import BedrockMeetingMinutesGenerator
from .intermediate import IntermediateTranscriptWriter
from .parser import VTTParser
from .preprocessor import TextPreprocessor

__all__ = [
    "BedrockMeetingMinutesGenerator",
    "IntermediateTranscriptWriter",
    "TextPreprocessor",
    "VTTParser",
]

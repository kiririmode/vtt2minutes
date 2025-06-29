"""VTT2Minutes - Convert Microsoft Teams VTT transcripts to structured meeting minutes."""

__version__ = "0.1.0"
__author__ = "VTT2Minutes Contributors"
__description__ = (
    "Convert Microsoft Teams VTT transcripts to structured meeting minutes"
)

from .parser import VTTParser
from .preprocessor import TextPreprocessor
from .summarizer import MeetingSummarizer

__all__ = ["MeetingSummarizer", "TextPreprocessor", "VTTParser"]

from enum import Enum
import json
from pathlib import Path
from typing import Optional, Dict, List, Union, Protocol

class OutputFormat(Enum):
    """Supported output formats for transcription conversion"""
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"

class FormatterProtocol(Protocol):
    """Protocol defining the interface for formatters"""
    def __init__(self, speaker_names: Optional[str] = None) -> None: ...
    def preamble(self) -> str: ...
    def format_chunk(self, chunk: Dict[str, Union[str, List[float]]], index: int) -> str: ...

class TxtFormatter:
    """Formats transcriptions as plain text"""
    def __init__(self, speaker_names: Optional[str] = None) -> None:
        self.speaker_names = speaker_names.split(',') if speaker_names else None
        self.speaker_map: Dict[str, str] = {}
        if self.speaker_names:
            self.speaker_map = {
                f"SPEAKER_{i:02}": name.upper().strip() 
                for i, name in enumerate(self.speaker_names)
            }

    def preamble(self) -> str:
        return ""

    def format_chunk(self, chunk: Dict[str, Union[str, List[float]]], index: int) -> str:
        text = chunk.get('text', '').strip()
        speaker = chunk.get('speaker', None)
        
        if speaker and self.speaker_names:
            speaker_name = self.speaker_map.get(speaker, speaker)
            return f"{speaker_name}: {text}\n"
        return f"{text}\n"

class SrtFormatter:
    """Formats transcriptions as SRT subtitles"""
    def __init__(self, speaker_names: Optional[str] = None) -> None:
        self.speaker_names = speaker_names.split(',') if speaker_names else None
        self.speaker_map: Dict[str, str] = {}
        if self.speaker_names:
            self.speaker_map = {
                f"SPEAKER_{i:02}": name.upper().strip() 
                for i, name in enumerate(self.speaker_names)
            }

    def preamble(self) -> str:
        return ""

    def format_seconds(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def format_chunk(self, chunk: Dict[str, Union[str, List[float]]], index: int) -> str:
        text = chunk.get('text', '').strip()
        speaker = chunk.get('speaker', None)
        timestamp = chunk['timestamp']
        start, end = float(timestamp[0]), float(timestamp[1])
        
        start_format = self.format_seconds(start)
        end_format = self.format_seconds(end)
        
        if speaker and self.speaker_names:
            speaker_name = self.speaker_map.get(speaker, speaker)
            text = f"{speaker_name}: {text}"
                
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

class VttFormatter:
    """Formats transcriptions as WebVTT subtitles"""
    def __init__(self, speaker_names: Optional[str] = None) -> None:
        self.speaker_names = speaker_names.split(',') if speaker_names else None
        self.speaker_map: Dict[str, str] = {}
        if self.speaker_names:
            self.speaker_map = {
                f"SPEAKER_{i:02}": name.upper().strip() 
                for i, name in enumerate(self.speaker_names)
            }

    def preamble(self) -> str:
        return "WEBVTT\n\n"

    def format_seconds(self, seconds: float) -> str:
        """Convert seconds to WebVTT timestamp format"""
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def format_chunk(self, chunk: Dict[str, Union[str, List[float]]], index: int) -> str:
        text = chunk.get('text', '').strip()
        speaker = chunk.get('speaker', None)
        timestamp = chunk['timestamp']
        start, end = float(timestamp[0]), float(timestamp[1])
        
        start_format = self.format_seconds(start)
        end_format = self.format_seconds(end)
        
        if speaker and self.speaker_names:
            speaker_name = self.speaker_map.get(speaker, speaker)
            text = f"{speaker_name}: {text}"
                
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

def convert_output(
    input_path: Path,
    output_format: OutputFormat,
    output_dir: Path,
    speaker_names: Optional[str] = None
) -> None:
    """
    Convert JSON transcription to specified format.
    
    Args:
        input_path: Path to input JSON file
        output_format: Format enum (OutputFormat.TXT, OutputFormat.SRT, or OutputFormat.VTT)
        output_dir: Output directory path
        verbose: Whether to print progress
        speaker_names: Comma-separated list of speaker names
    
    Raises:
        ValueError: If output_format is not a valid OutputFormat enum
        FileNotFoundError: If input_path doesn't exist or isn't readable
        json.JSONDecodeError: If input file isn't valid JSON
    """
    if not isinstance(output_format, OutputFormat):
        raise ValueError(f"output_format must be an OutputFormat enum, got {type(output_format)}")

    print(f"Converting {input_path} to {output_format.value} format...")

    with input_path.open('r', encoding='utf-8') as file:
        data = json.load(file)

    formatter: FormatterProtocol = {
        OutputFormat.TXT: TxtFormatter,
        OutputFormat.SRT: SrtFormatter,
        OutputFormat.VTT: VttFormatter
    }[output_format](speaker_names)

    output = formatter.preamble()
    chunks = data['chunks']

    for index, chunk in enumerate(chunks, 1):
        output += formatter.format_chunk(chunk, index)

    output_file_path = output_dir / f"{input_path.stem}.{output_format.value}"
    output_file_path.write_text(output, encoding='utf-8')

    print(f"Converted file saved at {output_file_path}")

import argparse

def output_format_type(value: str) -> OutputFormat:
    """
    Convert string to OutputFormat enum
    
    Args:
        value: String representation of the output format
    
    Returns:
        OutputFormat enum value
    
    Raises:
        argparse.ArgumentTypeError: If the format is not valid
    """
    try:
        value = value.lower()
        if value in [fmt.value for fmt in OutputFormat]:
            return OutputFormat(value)
        else:
            raise ValueError
    except ValueError:
        valid_formats = [f.value for f in OutputFormat]
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{value}'. Valid formats are: {', '.join(valid_formats)}"
        )
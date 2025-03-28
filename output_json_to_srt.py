#!/usr/bin/env python3
import json
import argparse
import os
import doctest
import sys
from typing import Dict, List, Any, Tuple


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    # Primero verificamos si se está ejecutando en modo test
    test_mode = "--test" in sys.argv
    
    parser = argparse.ArgumentParser(
        description="Convert JSON transcription file to SRT subtitle format"
    )
    
    # El archivo JSON solo es obligatorio si no estamos en modo test
    if test_mode:
        parser.add_argument(
            "json_file",
            type=str,
            nargs="?",  # Hace que el argumento sea opcional
            help="Path to the JSON transcription file (not needed with --test)"
        )
    else:
        parser.add_argument(
            "json_file",
            type=str,
            help="Path to the JSON transcription file"
        )
        
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output SRT file path (default: input filename with .srt extension)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run doctests instead of normal execution"
    )
    return parser.parse_args()


def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm)
    
    Examples:
    >>> seconds_to_srt_time(3661.5)
    '01:01:01,500'
    
    >>> seconds_to_srt_time(0)
    '00:00:00,000'
    
    >>> seconds_to_srt_time(59.999)
    '00:00:59,999'
    
    >>> seconds_to_srt_time(None)
    '00:00:00,000'
    
    >>> seconds_to_srt_time(3600)
    '01:00:00,000'
    
    >>> seconds_to_srt_time(7262.45)
    '02:01:02,450'
    """
    if seconds is None:  # Handle None values
        seconds = 0.0
        
    # Ensure we're working with a float
    seconds = float(seconds)
    
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    
    whole_seconds = int(seconds)
    # Usar redondeo para evitar errores de precisión en punto flotante
    milliseconds = round((seconds - whole_seconds) * 1000)
    
    # Si redondeamos a 1000 ms, ajustar los segundos
    if milliseconds == 1000:
        milliseconds = 0
        whole_seconds += 1
        # Propagar el desbordamiento si es necesario
        if whole_seconds == 60:
            whole_seconds = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def convert_json_to_srt(json_data: Dict[str, Any]) -> List[str]:
    """
    Convert JSON transcription data to SRT format
    
    Examples:
    >>> json_data = {
    ...     "speakers": [],
    ...     "chunks": [
    ...         {"timestamp": [0.0, 3.32], "text": "Hello world"},
    ...         {"timestamp": [3.64, 5.78], "text": "This is a test"}
    ...     ],
    ...     "text": "Hello world This is a test"
    ... }
    >>> convert_json_to_srt(json_data)
    ['1', '00:00:00,000 --> 00:00:03,320', 'Hello world', '', '2', '00:00:03,640 --> 00:00:05,780', 'This is a test', '']
    
    >>> # Test with missing or invalid timestamp
    >>> json_data = {
    ...     "chunks": [
    ...         {"timestamp": [None, 2.5], "text": "Missing start time"},
    ...         {"timestamp": [5.0, None], "text": "Missing end time"},
    ...         {"timestamp": [7.0, 9.0], "text": "Normal entry"}
    ...     ]
    ... }
    >>> result = convert_json_to_srt(json_data)
    >>> len(result)
    12
    >>> result[1]  # First entry start time is 0
    '00:00:00,000 --> 00:00:02,500'
    >>> result[5]  # Second entry end time is start time + 1
    '00:00:05,000 --> 00:00:06,000'
    
    >>> # Test with empty JSON
    >>> convert_json_to_srt({})
    []
    """
    srt_lines = []
    
    # Extract chunks from JSON
    chunks = json_data.get("chunks", [])
    
    for i, chunk in enumerate(chunks):
        # Get timestamps
        timestamps = chunk.get("timestamp", [0, 0])
        if len(timestamps) < 2 or timestamps[0] is None or timestamps[1] is None:
            # Skip invalid timestamps or set defaults
            start_time = 0 if timestamps[0] is None else timestamps[0]
            end_time = start_time + 1 if len(timestamps) < 2 or timestamps[1] is None else timestamps[1]
        else:
            start_time, end_time = timestamps
        
        # Format timestamps
        start_formatted = seconds_to_srt_time(start_time)
        end_formatted = seconds_to_srt_time(end_time)
        
        # Get text
        text = chunk.get("text", "").strip()
        
        # Add SRT entry
        srt_lines.append(f"{i+1}")
        srt_lines.append(f"{start_formatted} --> {end_formatted}")
        srt_lines.append(f"{text}")
        srt_lines.append("")  # Empty line between entries
    
    return srt_lines


def main() -> None:
    """
    Main function to handle the conversion process.
    """
    args = parse_arguments()
    
    # Run doctests if requested
    if args.test:
        failures, _ = doctest.testmod(verbose=True)
        if failures == 0:
            print("All tests passed!")
        return
    
    # Check if input file exists
    if not os.path.isfile(args.json_file):
        print(f"Error: Input file '{args.json_file}' not found.")
        return
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.json_file)[0]
        output_file = f"{base_name}.srt"
    
    try:
        # Read JSON file
        with open(args.json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert to SRT
        srt_lines = convert_json_to_srt(json_data)
        
        # Write SRT file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
        
        print(f"Successfully converted '{args.json_file}' to '{output_file}'")
        
    except json.JSONDecodeError:
        print(f"Error: '{args.json_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    main()
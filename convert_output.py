import argparse
import sys
from pathlib import Path
from formatters import OutputFormat, convert_output, output_format_type
from typing import Callable
import os

# Reemplazar el validador de archivos JSON con una función que solo verifique la extensión y los permisos
def json_file_validator(file_path: str) -> Path:
    """Create a validator for JSON files"""
    path = Path(file_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid file: {file_path}")
    if path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(f"Invalid JSON file: {file_path}")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"File is not readable: {file_path}")
    return path

# Reemplazar el validador de directorio escribible con una función que verifique si el directorio es escribible
def writable_dir_validator(dir_path: str) -> Path:
    """Check if the directory is writable"""
    path = Path(dir_path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid directory: {dir_path}")
    if not os.access(path, os.W_OK):
        raise argparse.ArgumentTypeError(f"Directory is not writable: {dir_path}")
    return path

def main():
    """Main function to handle command line arguments and conversion"""
    parser = argparse.ArgumentParser(
        description="Convert JSON transcription to specified format."
    )
    
    parser.add_argument(
        "file_name",
        type=json_file_validator,
        help="Path to the input JSON file to be converted.",
    )
    
    output_format_choices = [fmt.value for fmt in OutputFormat]
    parser.add_argument(
        "--output-format", "-out",
        type=str,
        default=OutputFormat.SRT.value,
        choices=output_format_choices,
        help=f"Output format for the transcription file ({', '.join(output_format_choices)})."
    )
    
    parser.add_argument(
        "--transcript-folder", "-f",
        type=writable_dir_validator,
        help="Folder to save the transcription output. Defaults to input file's directory."
    )
    
    parser.add_argument(
        "--speaker-names", "-sn",
        type=str,
        help="Comma-separated list of speaker names (e.g., 'Agustin,Sofia')."
    )

    args = parser.parse_args()
    
    # Si no se especifica la carpeta de salida, usar el directorio del archivo de entrada
    if args.transcript_folder is None:
        args.transcript_folder = args.file_name.parent

    # Imprimir el mapeo de los hablantes si se proporciona --speaker-names
    if args.speaker_names:
        speaker_names = args.speaker_names.split(',')
        speaker_map = {f"SPEAKER_{i:02}": name.upper().strip() for i, name in enumerate(speaker_names)}
        print("Speaker mapping:")
        for speaker_id, speaker_name in speaker_map.items():
            print(f"{speaker_id}: {speaker_name}")

    try:
        output_format = OutputFormat(args.output_format.lower())
        convert_output(
            input_path=args.file_name,
            output_format=output_format, 
            output_dir=args.transcript_folder,
            speaker_names=args.speaker_names
        )
    except Exception as e:
        print(f"Error processing file: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
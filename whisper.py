import json
import argparse
from transformers import pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import torch

# Add the `src` directory to PYTHONPATH
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from insanely_fast_whisper.utils.diarization_pipeline import diarize
from insanely_fast_whisper.utils.result import build_result

# Suppress specific warnings and configure logging level
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("rich").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
    parser.add_argument(
        "file_name",
        type=str,
        help="Path or URL to the audio file to be transcribed."
    )
    parser.add_argument(
        "--device-id",
        required=False,
        default="0",
        type=str,
        help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon.',
    )
    parser.add_argument(
        "--transcript-path",
        required=False,
        default="output.json",
        type=str,
        help="Path to save the transcription output.",
    )
    parser.add_argument(
        "--model-name",
        required=False,
        default="openai/whisper-large-v3",
        type=str,
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--task",
        required=False,
        default="transcribe",
        type=str,
        choices=["transcribe", "translate"],
        help="Task to perform: transcribe or translate.",
    )
    parser.add_argument(
        "--language",
        required=False,
        default="es",
        type=str,
        help="Language of the input audio.",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=8,
        help="Number of parallel batches to compute.",
    )
    parser.add_argument(
        "--hf-token",
        required=False,
        default="no_token",
        type=str,
        help="HuggingFace token for diarization.",
    )
    parser.add_argument(
        "--diarization_model",
        required=False,
        default="pyannote/speaker-diarization-3.1",
        type=str,
        help="Name of the pretrained model for diarization.",
    )
    # Añadir num_speakers
    parser.add_argument(
        "--num-speakers",
        required=False,
        default=None,
        type=int,
        help="Exact number of speakers (optional).",
    )
    parser.add_argument(
        "--min-speakers",
        required=False,
        default=None,
        type=int,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max-speakers",
        required=False,
        default=None,
        type=int,
        help="Maximum number of speakers.",
    )

    args = parser.parse_args()

    # Validar argumentos de speakers
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers cannot be used together with --min-speakers or --max-speakers")
    return args

if __name__ == "__main__":
    args = parse_arguments()

    # Configurar el pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device="cuda:" + args.device_id if args.device_id != "mps" else "mps",
        model_kwargs={"attn_implementation": "sdpa"},
    )

    # Configurar los parámetros de generación para español
    generate_kwargs = {
        "task": args.task,
        "language": args.language,
    }

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        outputs = pipe(
            args.file_name,
            chunk_length_s=30,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

    if args.hf_token != "no_token":
        speakers_transcript = diarize(args, outputs)
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed go check it out over here 👉 {args.transcript_path}"
        )
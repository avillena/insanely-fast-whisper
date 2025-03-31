#!/usr/bin/env python3
"""
Sistema de transcripción y diarización de audio con enfoque funcional.
Usa tipado estático específico y decoradores para importaciones dinámicas.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import time
import logging
import warnings
from pathlib import Path
from typing import Any, List, Union, cast

# Configuración del entorno
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

# Add these new filters
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", module="pyannote.audio.models.blocks.pooling")

# Filter SpeechBrain logs (they use the logging module)
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.WARNING)

# Importaciones locales
from data_types import (
    TranscriptionConfig, TaskType, FinalResult, 
    TranscriptOutput, DiarizedChunk
)
from helpers import log_time, logger
from audio import process_audio
from transcription import transcribe_audio
from diarization import diarize_audio
from formatters import OutputFormat, convert_output, output_format_type

"""
Modificación a la función parse_arguments() en transcribe.py para cambiar
la lógica de la ruta de salida predeterminada.
"""

@log_time
def parse_arguments() -> TranscriptionConfig:
    """
    Parsea los argumentos de línea de comandos y devuelve configuración tipada.
    
    Returns:
        TranscriptionConfig: Configuración inmutable con valores validados
    """
    parser = argparse.ArgumentParser(
        description="Sistema de transcripción y diarización de audio con enfoque funcional.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Transcripción básica de un archivo local
  transcribe.py archivo.mp3

  # Transcripción con formato SRT
  transcribe.py video.mp4 -f srt

  # Transcripción con diarización y nombres de hablantes
  transcribe.py reunion.wav --diarize --token=hf_xxx --speaker-names="Juan,María,Pedro"

  # Transcripción en inglés con atención optimizada
  transcribe.py entrevista.mp3 -l en --attn-type flash
        """
    )
    
    # Crear grupos de argumentos para mejor organización
    basic_group = parser.add_argument_group('opciones básicas')
    transc_group = parser.add_argument_group('opciones de transcripción')
    perf_group = parser.add_argument_group('opciones de rendimiento')
    diar_group = parser.add_argument_group('opciones de diarización')
    
    # Argumento posicional para archivo de entrada
    parser.add_argument(
        "file_name", 
        type=str, 
        help="Ruta o URL al archivo de audio/video a procesar"
    )
    
    # Opciones básicas
    basic_group.add_argument(
        "-o", "--output",
        dest="transcript_path",
        type=str,
        default=None,  # Cambiado de "output.json" a None para manejar dinámicamente
        help="Ruta para guardar el resultado (predeterminado: [nombre_input]-transcribe.[formato])"
    )
    
    basic_group.add_argument(
        "-f", "--format",
        dest="output_format",
        type=str,
        choices=["json", "srt", "vtt", "txt"],
        default="json",
        help="Formato del archivo de salida (predeterminado: json)"
    )
    
    # El resto del código del parser sigue igual
    # ...
    
    # Opciones de transcripción
    transc_group.add_argument(
        "-m", "--model",
        dest="model_name",
        type=str,
        default="openai/whisper-large-v3",
        help="Modelo de transcripción a utilizar (predeterminado: openai/whisper-large-v3)"
    )
    
    transc_group.add_argument(
        "-l", "--lang",
        dest="language",
        type=str,
        default="es",
        help="Idioma del audio (código ISO, p.ej. 'es', 'en', 'fr') (predeterminado: es)"
    )
    
    transc_group.add_argument(
        "-t", "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Tarea a realizar: transcribir o traducir a inglés (predeterminado: transcribe)"
    )
    
    transc_group.add_argument(
        "--chunk-length",
        type=int,
        default=30,
        help="Duración en segundos de cada fragmento de audio (predeterminado: 30)"
    )
    
    # Opciones de rendimiento
    perf_group.add_argument(
        "-d", "--device",
        dest="device_id",
        type=str,
        default="0",
        help="ID del dispositivo GPU (0, 1, etc.) o 'cpu' o 'mps' (predeterminado: 0)"
    )
    
    perf_group.add_argument(
        "-b", "--batch-size",
        type=int,
        default=8,
        help="Tamaño de lote para procesamiento por GPU (predeterminado: 8)"
    )
    
    perf_group.add_argument(
        "--attn-type",
        choices=["sdpa", "eager", "flash"],
        default="sdpa",
        help="Tipo de implementación de atención (predeterminado: sdpa)"
    )
    
    # Opciones de diarización
    diar_group.add_argument(
        "--diarize",
        action="store_true",
        help="Activar identificación de hablantes (requiere --token)"
    )
    
    diar_group.add_argument(
        "--dmodel",
        dest="diarization_model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Modelo de diarización a utilizar (predeterminado: pyannote/speaker-diarization-3.1)"
    )
    
    diar_group.add_argument(
        "--token",
        dest="hf_token",
        type=str,
        default="no_token",
        help="Token de HuggingFace para modelos de diarización"
    )
    
    # Grupo mutuamente excluyente para opciones de número de hablantes
    speakers_group = diar_group.add_mutually_exclusive_group()
    
    speakers_group.add_argument(
        "--num-speakers",
        type=int,
        help="Número exacto de hablantes en el audio"
    )
    
    speakers_group.add_argument(
        "--min-speakers",
        type=int,
        help="Número mínimo de hablantes a detectar"
    )
    
    diar_group.add_argument(
        "--max-speakers",
        type=int,
        help="Número máximo de hablantes a detectar"
    )
    
    diar_group.add_argument(
        "--speaker-names",
        type=str,
        help="Lista de nombres separados por comas para reemplazar etiquetas de hablantes (ej: \"Juan,María,Pedro\")"
    )
    
    args = parser.parse_args()
    
    # Validaciones adicionales
    if args.max_speakers is not None and args.min_speakers is None:
        parser.error("--max-speakers requiere --min-speakers")
    
    if args.diarize and args.hf_token == "no_token":
        parser.error("La opción --diarize requiere un token de HuggingFace (--token)")
    
    # NUEVO: Calcular la ruta de salida predeterminada si no se especificó
    if args.transcript_path is None:
        # Obtener la ruta completa del archivo de entrada
        input_path = Path(args.file_name)
        
        # Si es una URL, usar solo el nombre del archivo
        if args.file_name.startswith(("http://", "https://")):
            import urllib.parse
            file_name = urllib.parse.urlparse(args.file_name).path.split("/")[-1]
            input_path = Path(file_name)
        
        # Obtener el directorio y el nombre base del archivo de entrada
        input_dir = input_path.parent
        input_stem = input_path.stem
        
        # Generar el nombre del archivo de salida
        output_name = f"{input_stem}-transcribe.{args.output_format}"
        
        # Combinar con el directorio para obtener la ruta completa
        args.transcript_path = str(input_dir / output_name)
        logger.info(f"Ruta de salida predeterminada: [bold cyan]{args.transcript_path}[/]")
    
    # Ajustar la extensión del archivo de salida según el formato
    if args.output_format != "json":
        output_path = Path(args.transcript_path)
        # Si la extensión no coincide con el formato, cambiarla
        if output_path.suffix.lower() != f".{args.output_format}":
            stem = output_path.stem
            args.transcript_path = str(output_path.parent / f"{stem}.{args.output_format}")
    
    # Crear configuración tipada y validada
    return TranscriptionConfig(
        file_name=args.file_name,
        device_id=args.device_id,
        transcript_path=args.transcript_path,
        model_name=args.model_name,
        task=cast(TaskType, args.task),
        language=args.language,
        batch_size=args.batch_size,
        output_format=args.output_format,
        chunk_length=args.chunk_length,
        attn_type=args.attn_type,
        hf_token=args.hf_token,
        diarization_model=args.diarization_model,
        diarize=args.diarize,
        speaker_names=args.speaker_names,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )

@log_time
def build_and_save_result(
    config: TranscriptionConfig, 
    transcript: TranscriptOutput, 
    speakers_transcript: List[DiarizedChunk]
) -> FinalResult:
    """
    Construye y guarda el resultado final en formato JSON y lo convierte al formato solicitado.
    
    Args:
        config: Configuración de transcripción
        transcript: Resultados de la transcripción
        speakers_transcript: Transcripción con info de hablantes
        
    Returns:
        FinalResult: Resultado final con toda la información
    """
    # Construir resultado final
    logger.info("Construyendo resultado final")
    result: FinalResult = {
        "speakers": speakers_transcript,
        "chunks": transcript["chunks"],
        "text": transcript["text"],
    }
    
    # Guardar resultado en JSON primero (siempre necesario para la conversión)
    json_output_path = Path(str(config.transcript_path).replace(f".{config.output_format}", ".json")) \
        if config.output_format != "json" else Path(config.transcript_path)
    
    logger.info(f"Guardando resultado JSON en: [bold cyan]{json_output_path}[/]")
    with open(json_output_path, "w", encoding="utf8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    
    # Si el formato solicitado no es JSON, convertir
    if config.output_format != "json":
        try:
            output_format = OutputFormat(config.output_format)
            convert_output(
                input_path=json_output_path,
                output_format=output_format,
                output_dir=json_output_path.parent,
                speaker_names=config.speaker_names
            )
            
            # Si el formato de salida no es JSON, eliminar el archivo JSON temporal 
            # solo si el archivo de salida es diferente
            if str(config.transcript_path) != str(json_output_path):
                os.remove(json_output_path)
                logger.info(f"Archivo JSON temporal eliminado: [bold cyan]{json_output_path}[/]")
        except Exception as e:
            logger.error(f"[bold red]Error al convertir formato[/]: {str(e)}")
            logger.info(f"Se mantiene el resultado en formato JSON: [bold cyan]{json_output_path}[/]")
    else:
        logger.info(f"[bold green]¡Voila!✨[/] Archivo guardado en: [bold cyan]{json_output_path}[/]")
    
    return result

@log_time
def main() -> FinalResult:
    """
    Función principal que coordina todo el proceso.
    
    Returns:
        FinalResult: Resultado final del proceso
    """
    try:
        # 1. Procesar argumentos
        config: TranscriptionConfig = parse_arguments()
        
        # 2. Procesar audio
        audio_data = process_audio(config.file_name)
        
        # 3. Transcribir audio
        transcript = transcribe_audio(config, audio_data)
        
        # 4. Diarizar audio (opcional)
        speakers_transcript = []
        if config.diarize:
            speakers_transcript = diarize_audio(config, audio_data, transcript)
        
        # 5. Construir y guardar resultado
        result = build_and_save_result(config, transcript, speakers_transcript)
        
        return result
    except Exception as e:
        logger.error(f"[bold red]Error durante la ejecución:[/] {str(e)}")
        logger.debug("Detalles del error:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

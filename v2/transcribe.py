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

# Importaciones locales
from data_types import (
    TranscriptionConfig, TaskType, FinalResult, 
    TranscriptOutput, DiarizedChunk
)
from helpers import log_time, logger
from audio import process_audio
from transcription import transcribe_audio
from diarization import diarize_audio

@log_time
def parse_arguments() -> TranscriptionConfig:
    """
    Parsea los argumentos de línea de comandos y devuelve configuración tipada.
    
    Returns:
        TranscriptionConfig: Configuración inmutable con valores validados
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Transcripción de audio")
    parser.add_argument(
        "file_name", type=str, help="Ruta o URL al archivo de audio"
    )
    parser.add_argument(
        "--device-id", default="0", type=str, help="ID del dispositivo GPU"
    )
    parser.add_argument(
        "--transcript-path", default="output.json", type=str, help="Ruta de salida"
    )
    parser.add_argument(
        "--model-name", default="openai/whisper-large-v3", type=str
    )
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"], 
        type=str, help="Tarea a realizar"
    )
    parser.add_argument(
        "--language", default="es", type=str, help="Idioma del audio"
    )
    parser.add_argument(
        "--batch-size", default=8, type=int, help="Tamaño de lote"
    )
    parser.add_argument(
        "--hf-token", default="no_token", type=str, help="Token de HuggingFace"
    )
    parser.add_argument(
        "--diarization-model", default="pyannote/speaker-diarization-3.1", type=str
    )
    parser.add_argument(
        "--num-speakers", default=None, type=int, help="Número exacto de hablantes"
    )
    parser.add_argument(
        "--min-speakers", default=None, type=int, help="Número mínimo de hablantes"
    )
    parser.add_argument(
        "--max-speakers", default=None, type=int, help="Número máximo de hablantes"
    )

    args: argparse.Namespace = parser.parse_args()
    
    # Crear configuración tipada y validada
    return TranscriptionConfig(
        file_name=args.file_name,
        device_id=args.device_id,
        transcript_path=args.transcript_path,
        model_name=args.model_name,
        task=cast(TaskType, args.task),  # Cast para Literal
        language=args.language,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        diarization_model=args.diarization_model,
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
    Construye y guarda el resultado final.
    
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
    
    # Guardar resultado
    output_path: Union[str, Path] = config.transcript_path
    logger.info(f"Guardando resultado en {output_path}")
    with open(output_path, "w", encoding="utf8") as fp:
        json.dump(result, fp, ensure_ascii=False)
    
    logger.info(f"Voila!✨ Archivo guardado en: {output_path}")
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
        speakers_transcript = diarize_audio(config, audio_data, transcript)
        
        # 5. Construir y guardar resultado
        result = build_and_save_result(config, transcript, speakers_transcript)
        
        return result
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

"""
Funciones para la diarización (identificación de hablantes) de audio.
"""
import sys
from typing import Dict, Any, List, Tuple, cast

from helpers import log_time, with_imports, with_progress_bar, logger, format_path
from data_types import (
    TranscriptionConfig, AudioData, TranscriptOutput, 
    TranscriptChunk, DiarizedChunk
)

@with_imports("torch", "pyannote.audio")
def load_diarization_pipeline(
    config: TranscriptionConfig,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> Any:
    """
    Carga el pipeline de diarización.
    
    Args:
        config: Configuración de transcripción/diarización
        dynamic_imports: Módulos importados dinámicamente
        
    Returns:
        Pipeline: Pipeline de diarización cargado
    """
    torch = dynamic_imports["torch"]
    pyannote_audio = dynamic_imports["audio"]
    
    pipeline = pyannote_audio.Pipeline.from_pretrained(
        checkpoint_path=config.diarization_model,
        use_auth_token=config.hf_token,
    )
    
    device = torch.device("mps" if config.device_id == "mps" 
                         else "cpu" if config.device_id == "cpu" 
                         else f"cuda:{config.device_id}")
    pipeline.to(device)
    
    return pipeline

def process_diarization_segments(diarization: Any) -> List[Dict[str, Any]]:
    """
    Procesa los segmentos de diarización y los combina por hablante.
    
    Args:
        diarization: Resultado de la diarización
        
    Returns:
        List[Dict[str, Any]]: Segmentos combinados por hablante
    """
    # Extraer segmentos
    segments: List[Dict[str, Any]] = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({
            "segment": {"start": segment.start, "end": segment.end},
            "track": track,
            "label": label,
        })
    
    # Combinar segmentos consecutivos del mismo hablante
    new_segments: List[Dict[str, Any]] = []
    if segments:
        prev_segment: Dict[str, Any] = segments[0]
        
        for i in range(1, len(segments)):
            cur_segment: Dict[str, Any] = segments[i]
            
            # Si cambió el hablante, agregar el segmento combinado
            if cur_segment["label"] != prev_segment["label"]:
                new_segments.append({
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                })
                prev_segment = segments[i]
        
        # Agregar el último segmento
        new_segments.append({
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": segments[-1]["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        })
    
    return new_segments

@with_imports("numpy")
def align_segments_with_transcript(
    new_segments: List[Dict[str, Any]], 
    transcript_chunks: List[TranscriptChunk],
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> List[DiarizedChunk]:
    """
    Alinea los segmentos de diarización con la transcripción.
    
    Args:
        new_segments: Segmentos de diarización
        transcript_chunks: Fragmentos de transcripción
        dynamic_imports: Módulos importados dinámicamente
        
    Returns:
        List[DiarizedChunk]: Fragmentos con información de hablante
    """
    np = dynamic_imports["numpy"]
    segmented_preds: List[DiarizedChunk] = []
    
    if not new_segments or not transcript_chunks:
        return segmented_preds
    
    # Obtener timestamps finales de cada fragmento transcrito
    end_timestamps = np.array([
        chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None 
        else sys.float_info.max for chunk in transcript_chunks
    ])
    
    # Alinear los timestamps de diarización y ASR
    for segment in new_segments:
        end_time: float = segment["segment"]["end"]
        # Encontrar el timestamp de ASR más cercano
        upto_idx: int = np.argmin(np.abs(end_timestamps - end_time))
        
        # Agregar fragmentos con información de hablante
        for i in range(upto_idx + 1):
            chunk: TranscriptChunk = transcript_chunks[i]
            segmented_preds.append({
                "text": chunk["text"],
                "timestamp": chunk["timestamp"],
                "speaker": segment["speaker"]
            })
        
        # Recortar la transcripción y timestamps para el siguiente segmento
        transcript_chunks = transcript_chunks[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]
        
        if len(end_timestamps) == 0:
            break
    
    return segmented_preds

@log_time
@with_imports("torch", "pyannote.audio", "numpy")
def diarize_audio(
    config: TranscriptionConfig, 
    audio_data: AudioData, 
    transcript: TranscriptOutput,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> List[DiarizedChunk]:
    """
    Realiza la diarización para identificar diferentes hablantes.
    
    Args:
        config: Configuración de transcripción/diarización
        audio_data: Audio procesado para diarización
        transcript: Resultados de la transcripción
        dynamic_imports: Módulos importados dinámicamente
        
    Returns:
        List[DiarizedChunk]: Transcripción con información de hablantes
    """
    # Si no hay token, omitir diarización
    if config.hf_token == "no_token":
        logger.info("Diarización omitida (no se proporcionó token)")
        return []
    
    logger.info(f"Iniciando diarización con modelo {config.diarization_model}")
    
    # Mostrar información de la fuente si está disponible
    if "source_info" in audio_data and audio_data["source_info"].get("path"):
        logger.info(f"Fuente: {format_path(audio_data['source_info']['path'])}")
    
    def execute_diarization() -> Tuple[List[DiarizedChunk], List[Dict[str, Any]]]:
        # 1. Cargar pipeline
        diarization_pipeline = load_diarization_pipeline(
            config, dynamic_imports={"torch": dynamic_imports["torch"], 
                                    "pyannote.audio": dynamic_imports["audio"]}
        )
        
        # 2. Ejecutar diarización
        diarization = diarization_pipeline(
            {"waveform": audio_data["torch_tensor"], "sample_rate": 16000},
            num_speakers=config.num_speakers,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )
        
        # 3. Procesar segmentos
        new_segments: List[Dict[str, Any]] = process_diarization_segments(diarization)
        
        # 4. Alinear con transcripción
        transcript_chunks: List[TranscriptChunk] = transcript["chunks"].copy()
        segmented_preds: List[DiarizedChunk] = align_segments_with_transcript(
            new_segments, transcript_chunks, 
            dynamic_imports={"numpy": dynamic_imports["numpy"]}
        )
        
        return segmented_preds, new_segments
    
    segmented_preds, new_segments = with_progress_bar("Segmentando hablantes...", execute_diarization)
    
    num_speakers: int = len(set(segment["speaker"] for segment in new_segments)) if new_segments else 0
    logger.info(f"Diarización completa: {num_speakers} hablantes detectados")
    
    return segmented_preds

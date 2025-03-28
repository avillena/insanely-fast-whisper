"""
Funciones para la transcripción de audio.
"""
from typing import Dict, Any, cast

from helpers import log_time, with_imports, with_progress_bar, logger
from data_types import TranscriptionConfig, AudioData, TranscriptOutput

@log_time
@with_imports("torch", "transformers")
def transcribe_audio(
    config: TranscriptionConfig, 
    audio_data: AudioData,
    *, 
    dynamic_imports: Dict[str, Any] = {}
) -> TranscriptOutput:
    """
    Transcribe el audio usando el modelo Whisper.

    Args:
        config: Configuración de transcripción.
        audio_data: Audio procesado, conteniendo "numpy_array" y "sampling_rate".
        dynamic_imports: Módulos importados dinámicamente.

    Returns:
        TranscriptOutput: Resultado de la transcripción.
    """
    logger.info(f"Iniciando transcripción con modelo {config.model_name}")
    
    torch = dynamic_imports["torch"]
    transformers = dynamic_imports["transformers"]
    
    # Configurar el pipeline de ASR
    model_kwargs = {}
    
    # Configurar el tipo de atención
    if config.attn_type == "sdpa":
        model_kwargs["attn_implementation"] = "sdpa"
    elif config.attn_type == "eager":
        model_kwargs["attn_implementation"] = "eager" 
    elif config.attn_type == "flash":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=config.model_name,
        torch_dtype=torch.float16,
        device="cuda:" + config.device_id if config.device_id != "mps" else "mps",
        model_kwargs=model_kwargs,
    )
    
    # Parámetros de generación
    generate_kwargs: Dict[str, str] = {
        "task": config.task,
        "language": config.language,
    }
    
    def execute_transcription() -> Any:
        # Construir el diccionario de entrada según lo que espera el pipeline
        audio_dict = {
            "raw": audio_data["numpy_array"].copy(),
            "sampling_rate": audio_data["sampling_rate"],
        }
        
        return pipe(
            audio_dict,
            chunk_length_s=config.chunk_length,
            batch_size=config.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )
    
    outputs: Any = with_progress_bar("Transcribiendo...", execute_transcription)
    logger.info(f"Transcripción completa: {len(outputs.get('chunks', []))} fragmentos")
    return cast(TranscriptOutput, outputs)

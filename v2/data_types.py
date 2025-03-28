"""
Definiciones de tipos de datos para el sistema de transcripción.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict, Dict, List, Any, Union, Optional, Tuple, Literal
from pathlib import Path

# Tipos específicos para datos
TaskType = Literal["transcribe", "translate"]

class AudioData(TypedDict):
    """Datos de audio procesados y listos para transcripción/diarización."""
    numpy_array: "np.ndarray"  # Array de audio mono de 16kHz
    torch_tensor: "torch.Tensor"  # Tensor para diarización
    sampling_rate: int  # Siempre 16000Hz

class TranscriptChunk(TypedDict):
    """Fragmento individual de transcripción con timestamp."""
    text: str
    timestamp: Tuple[Optional[float], Optional[float]]

class TranscriptOutput(TypedDict):
    """Resultado de la transcripción completa."""
    text: str  # Texto completo
    chunks: List[TranscriptChunk]  # Fragmentos con timestamps

class SpeakerSegment(TypedDict):
    """Segmento con información de hablante."""
    segment: Dict[str, float]  # start, end en segundos
    speaker: str  # Identificador del hablante (SPEAKER_00, etc.)

class DiarizedChunk(TranscriptChunk):
    """Fragmento de transcripción con información de hablante."""
    speaker: str  # Identificador del hablante

class FinalResult(TypedDict):
    """Estructura final del resultado."""
    speakers: List[DiarizedChunk]  # Transcripción con hablantes
    chunks: List[TranscriptChunk]  # Fragmentos originales
    text: str  # Texto completo

@dataclass(frozen=True)
class TranscriptionConfig:
    """Configuración inmutable para el proceso de transcripción."""
    file_name: Union[str, Path]
    device_id: str = "0"
    transcript_path: Union[str, Path] = "output.json"
    model_name: str = "openai/whisper-large-v3"
    task: TaskType = "transcribe"
    language: str = "es"
    batch_size: int = 8
    output_format: str = "json"
    chunk_length: int = 30
    attn_type: str = "sdpa"
    hf_token: str = "no_token"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarize: bool = False
    speaker_names: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validar configuración después de inicialización."""
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError(
                "--num-speakers no puede usarse junto con --min-speakers o --max-speakers"
            )

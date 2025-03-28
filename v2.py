"""
Sistema de transcripción y diarización de audio con enfoque funcional.
Usa tipado estático específico y decoradores para importaciones dinámicas.
"""
from __future__ import annotations
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import time
import logging
import functools
import json
import sys
import argparse
import importlib
import requests
from pathlib import Path
from typing import TypedDict, Dict, List, Any, Union, Optional, Callable, Tuple, cast, Literal
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración de logging con formato compacto
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s-%(name)s-%(levelname)s - %(message)s',
                    datefmt='%y%m%d.%H%M%S')
logger = logging.getLogger("transcriber")

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

# Configuración usando dataclass congelado
from dataclasses import dataclass

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
    hf_token: str = "no_token"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
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

# Utilidad para importación dinámica
def import_module(module_name: str) -> Any:
    """Importa un módulo dinámicamente y registra en el log."""
    logger.debug(f"Importando: {module_name}")
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Error importando {module_name}: {e}")
        raise

# Decorador para importar módulos justo antes de ejecutar la función
def with_imports(*module_names: str) -> Callable:
    """
    Decorador que importa módulos justo antes de ejecutar la función.
    
    Args:
        *module_names: Nombres de los módulos a importar
    
    Returns:
        Función decorada que tendrá los módulos importados disponibles
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dynamic_imports: Dict[str, Any] = {}
            for name in module_names:
                module_alias = name.split(".")[-1]
                dynamic_imports[module_alias] = import_module(name)
            kwargs["dynamic_imports"] = dynamic_imports
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Decorador para medir tiempo de ejecución
def log_time(func: Callable) -> Callable:
    """Decorador que mide el tiempo de ejecución de una función."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        logger.info(f"{func.__name__} - Tiempo: {end_time - start_time:.2f}s")
        return result
    return wrapper

# Función auxiliar para mostrar barra de progreso
def with_progress_bar(description: str, func: Callable) -> Any:
    """
    Ejecuta una función mostrando una barra de progreso.
    
    Args:
        description: Descripción para la barra de progreso
        func: Función a ejecutar
        
    Returns:
        Resultado de la función
    """
    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"[yellow]{description}", total=None)
        return func()

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
@with_imports("numpy", "torch", "transformers.pipelines.audio_utils", "torchaudio.functional", "os", "subprocess")
def process_audio(
    input_src: Union[str, Path, bytes, Dict[str, Any]],
    *,
    dynamic_imports: Dict[str, Any] = {}
) -> AudioData:
    """
    Procesa el audio desde varias fuentes para ASR y diarización.
    Detecta automáticamente archivos de video y extrae el audio.
    
    Args:
        input_src: String (ruta/URL), bytes o diccionario con datos de audio.
        dynamic_imports: Módulos importados dinámicamente.
        
    Returns:
        AudioData: Diccionario con numpy_array, torch_tensor y sampling_rate.
    """
    logger.info(f"Procesando audio desde {type(input_src)}")
    
    # Extraer módulos
    np = dynamic_imports["numpy"]
    torch = dynamic_imports["torch"]
    audio_utils = dynamic_imports["audio_utils"]
    functional = dynamic_imports["functional"]
    os = dynamic_imports["os"]
    subprocess = dynamic_imports["subprocess"]

    # Procesamiento según tipo de entrada
    if isinstance(input_src, (str, Path)):
        input_str: str = str(input_src)
        _, file_ext = os.path.splitext(input_str.lower())
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if input_str.startswith(("http://", "https://")):
            logger.info("Descargando audio desde URL")
            input_src = requests.get(input_str).content
        else:
            logger.info("Cargando archivo desde local")
            if is_video:
                logger.info(f"Detectado archivo de video: {file_ext}")
                temp_audio = f"{os.path.splitext(input_str)[0]}_temp.wav"
                logger.info(f"Extrayendo audio a archivo temporal: {temp_audio}")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", input_str, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio],
                        check=True,
                        capture_output=True
                    )
                    # Verificar que el archivo se generó correctamente
                    if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                        raise ValueError("El archivo temporal de audio no se generó correctamente o está vacío")
                    with open(temp_audio, "rb") as f:
                        input_src = f.read()
                    os.remove(temp_audio)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error al extraer audio con FFmpeg: {e}")
                    logger.error(f"Salida de error: {e.stderr.decode() if e.stderr else 'No stderr'}")
                    raise ValueError(f"No se pudo extraer audio del archivo de video: {input_str}") from e
                except FileNotFoundError:
                    logger.error("FFmpeg no está instalado o no se encuentra en el PATH")
                    raise ValueError("FFmpeg es necesario para procesar archivos de video")
            else:
                with open(input_str, "rb") as f:
                    input_src = f.read()

    if isinstance(input_src, bytes):
        logger.info("Decodificando audio con FFmpeg")
        try:
            input_src = audio_utils.ffmpeg_read(input_src, 16000)
        except Exception as e:
            logger.error("Error al decodificar el audio con FFmpeg", exc_info=True)
            raise ValueError("No se pudo decodificar el audio correctamente. Verifica el formato del archivo.") from e

    if isinstance(input_src, dict):
        logger.info("Procesando entrada tipo diccionario")
        if not ("sampling_rate" in input_src and ("raw" in input_src or "array" in input_src)):
            raise ValueError("El diccionario debe contener 'raw' o 'array' con el audio y 'sampling_rate'")
        _inputs: Optional[Any] = input_src.pop("raw", None)
        if _inputs is None:
            input_src.pop("path", None)
            _inputs = input_src.pop("array", None)
        in_sampling_rate: int = input_src.pop("sampling_rate")
        input_src = _inputs
        if in_sampling_rate != 16000:
            logger.info(f"Remuestreando de {in_sampling_rate} a 16000 Hz")
            input_src = functional.resample(torch.from_numpy(input_src), in_sampling_rate, 16000).numpy()

    if not isinstance(input_src, np.ndarray):
        raise ValueError(f"Se esperaba un array numpy, se obtuvo `{type(input_src)}`")
    if len(input_src.shape) != 1:
        raise ValueError("Se esperaba audio de un solo canal")

    # Preparar tensor para la diarización
    input_src = input_src.copy() 
    torch_tensor = torch.from_numpy(input_src).float().unsqueeze(0)
    
    logger.info(f"Audio procesado: forma={input_src.shape}, SR=16000Hz")
    return {
        "numpy_array": input_src,
        "torch_tensor": torch_tensor,
        "sampling_rate": 16000
    }


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
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=config.model_name,
        torch_dtype=torch.float16,
        device="cuda:" + config.device_id if config.device_id != "mps" else "mps",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    
    # Parámetros de generación
    generate_kwargs: Dict[str, str] = {
        "task": config.task,
        "language": config.language,
    }
    
    def execute_transcription() -> Any:
        # Construir el diccionario de entrada según lo que espera el pipeline:
        # Debe contener una clave "raw" con el array de audio y "sampling_rate"
        audio_dict = {
            "raw": audio_data["numpy_array"].copy(),  # Hacemos copy() para que sea escribible
            "sampling_rate": audio_data["sampling_rate"],
        }
        
        return pipe(
            audio_dict,
            chunk_length_s=30,
            batch_size=config.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )
    
    outputs: Any = with_progress_bar("Transcribiendo...", execute_transcription)
    logger.info(f"Transcripción completa: {len(outputs.get('chunks', []))} fragmentos")
    return cast(TranscriptOutput, outputs)




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
    pyannote_audio = dynamic_imports["pyannote.audio"]
    
    pipeline = pyannote_audio.Pipeline.from_pretrained(
        checkpoint_path=config.diarization_model,
        use_auth_token=config.hf_token,
    )
    
    device = torch.device("mps" if config.device_id == "mps" else f"cuda:{config.device_id}")
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
    
    def execute_diarization() -> Tuple[List[DiarizedChunk], List[Dict[str, Any]]]:
        # 1. Cargar pipeline
        diarization_pipeline = load_diarization_pipeline(
            config, dynamic_imports={"torch": dynamic_imports["torch"], 
                                    "pyannote.audio": dynamic_imports["pyannote.audio"]}
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
        
        # 2. Procesar audio - usa el decorador @with_imports
        audio_data: AudioData = process_audio(config.file_name)
        
        # 3. Transcribir audio - usa el decorador @with_imports
        transcript: TranscriptOutput = transcribe_audio(config, audio_data)
        
        # 4. Diarizar audio (opcional) - usa el decorador @with_imports
        speakers_transcript: List[DiarizedChunk] = diarize_audio(config, audio_data, transcript)
        
        # 5. Construir y guardar resultado
        result: FinalResult = build_and_save_result(config, transcript, speakers_transcript)
        
        return result
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
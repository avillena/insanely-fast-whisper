"""
Funciones para procesamiento de audio.
"""
import os
import requests
from pathlib import Path
from typing import Dict, Any, Union

from helpers import log_time, with_imports, logger
from data_types import AudioData

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
        _inputs: Any = input_src.pop("raw", None)
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

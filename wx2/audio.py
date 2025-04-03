"""
Funciones para procesamiento de audio.
"""
import os
import requests
from pathlib import Path
from typing import Dict, Any, Union

from helpers import log_time, with_imports, logger, format_path
from data_types import AudioData, AudioSourceInfo

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
    # Inicializar información de origen
    source_info: AudioSourceInfo = {
        "path": None,
        "type": "unknown",
        "file_name": None,
        "format": "unknown",
        "is_video": False,
        "duration_seconds": None,
        "content_size": None
    }
    
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
        source_info["path"] = input_str
        source_info["file_name"] = os.path.basename(input_str)
        
        _, file_ext = os.path.splitext(input_str.lower())
        source_info["format"] = file_ext.lstrip('.')
        source_info["is_video"] = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if input_str.startswith(("http://", "https://")):
            source_info["type"] = "url"
            logger.info(f"Procesando audio desde URL: {format_path(input_str)} ({'' if source_info['is_video'] else 'audio, '}{source_info['format']})")
            logger.info("Descargando audio desde URL")
            
            response = requests.get(input_str, stream=True)
            content_size = int(response.headers.get('content-length', 0))
            source_info["content_size"] = content_size
            
            input_src = response.content
            logger.info(f"Descarga completada: {content_size / (1024*1024):.1f} MB recibidos")
        else:
            source_info["type"] = "file"
            logger.info(f"Procesando audio desde archivo: {format_path(input_str)} ({source_info['format']})")
            logger.info("Cargando archivo desde local")
            
            if source_info["is_video"]:
                logger.info(f"Detectado archivo de video: {file_ext}")
                temp_audio = f"{os.path.splitext(input_str)[0]}_temp.wav"
                logger.info(f"Extrayendo audio a archivo temporal: {format_path(temp_audio)}")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", input_str, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio],
                        check=True,
                        capture_output=True
                    )
                    # Verificar que el archivo se generó correctamente
                    if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                        raise ValueError("El archivo temporal de audio no se generó correctamente o está vacío")
                    
                    logger.info(f"Audio extraído correctamente, tamaño: {os.path.getsize(temp_audio)/1024:.1f} KB")
                    with open(temp_audio, "rb") as f:
                        input_src = f.read()
                        logger.info(f"Bytes leídos del archivo temporal: {len(input_src)} bytes")
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

    # Procesamiento de datos binarios con manejo robusto de errores
    if isinstance(input_src, bytes):
        source_info["type"] = "bytes"
        logger.info(f"Procesando audio desde datos binarios: {len(input_src) / 1024:.1f} KB")
        source_info["content_size"] = len(input_src)
        logger.info("Decodificando audio con FFmpeg")
        
        try:
            # Añadir logging detallado para diagnóstico
            logger.info(f"Tamaño de los bytes antes de FFmpeg: {len(input_src)} bytes")
            decoded_audio = audio_utils.ffmpeg_read(input_src, 16000)
            
            # Verificar explícitamente que se decodificó correctamente
            if not isinstance(decoded_audio, np.ndarray):
                logger.error(f"FFmpeg no devolvió un array numpy, sino {type(decoded_audio).__name__}")
                raise ValueError(f"Decodificación incorrecta: se esperaba numpy.ndarray, se obtuvo {type(decoded_audio).__name__}")
                
            input_src = decoded_audio
            logger.info(f"Audio decodificado correctamente: {input_src.shape[0]} muestras")
            
        except Exception as e:
            logger.error(f"Error al decodificar el audio con FFmpeg: {str(e)}", exc_info=True)
            raise ValueError("No se pudo decodificar el audio correctamente. Verifica el formato del archivo y la instalación de FFmpeg.") from e

    elif isinstance(input_src, dict):
        source_info["type"] = "dict"
        if "path" in input_src:
            source_info["path"] = input_src["path"]
            source_info["file_name"] = os.path.basename(str(input_src["path"]))
        
        logger.info(f"Procesando audio desde diccionario: {source_info.get('path', 'datos en memoria')}")
        
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

    # Verificación final del tipo de datos
    if not isinstance(input_src, np.ndarray):
        error_msg = f"Se esperaba un array numpy, se obtuvo `{type(input_src).__name__}`"
        logger.error(error_msg)
        if isinstance(input_src, bytes):
            logger.error(f"Los datos siguen siendo bytes de tamaño {len(input_src)}. La decodificación de FFmpeg falló.")
        raise ValueError(error_msg)
        
    if len(input_src.shape) != 1:
        raise ValueError("Se esperaba audio de un solo canal")

    # Preparar tensor para la diarización
    input_src = input_src.copy() 
    torch_tensor = torch.from_numpy(input_src).float().unsqueeze(0)
    
    # Calcular duración
    duration_seconds = len(input_src) / 16000
    source_info["duration_seconds"] = duration_seconds
    duration_str = f"{int(duration_seconds//60)}m {int(duration_seconds%60)}s"
    
    logger.info(f"Audio procesado: {input_src.shape[0]} muestras, SR=16000Hz (duración: {duration_str})")
    
    return {
        "numpy_array": input_src,
        "torch_tensor": torch_tensor,
        "sampling_rate": 16000,
        "source_info": source_info
    }
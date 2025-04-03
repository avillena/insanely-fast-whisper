"""
Módulo para convertir transcripciones a diferentes formatos.
Versión optimizada para eliminar duplicación de código.
"""
import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

from helpers import logger, format_path

class OutputFormat(Enum):
    """Formatos de salida soportados para transcripciones."""
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"

def output_format_type(value: str) -> OutputFormat:
    """Convertir string a enum OutputFormat."""
    try:
        return OutputFormat(value.lower())
    except ValueError:
        raise ValueError(f"Formato no soportado: {value}")

def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """
    Formatea segundos a formato de timestamp.
    
    Args:
        seconds: Tiempo en segundos
        format_type: Tipo de formato ("srt" o "vtt")
        
    Returns:
        str: Timestamp formateado
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    if format_type == "vtt":
        return f"{hours:02}:{minutes:02}:{secs:02}.{millisecs:03}"
    else:  # srt por defecto
        return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def get_speaker_display(segment: Dict[str, Any], speaker_names: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Obtiene el nombre de visualización del hablante.
    
    Args:
        segment: Segmento de transcripción
        speaker_names: Mapeo de ID de hablante a nombre personalizado
        
    Returns:
        Optional[str]: Nombre de visualización del hablante o None si no hay información
    """
    if 'speaker' not in segment:
        return None
        
    speaker_id = segment['speaker']
    if speaker_names:
        return speaker_names.get(speaker_id, speaker_id)
    return speaker_id

def get_segments_from_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extrae los segmentos de la transcripción.
    
    Args:
        data: Datos de transcripción
        
    Returns:
        List[Dict[str, Any]]: Lista de segmentos
    """
    # Usar 'speakers' si está disponible, de lo contrario 'chunks'
    return data.get('speakers', data.get('chunks', []))

def format_subtitle_content(
    segments: List[Dict[str, Any]],
    speaker_names: Optional[Dict[str, str]],
    format_type: str
) -> str:
    """
    Formatea una lista de segmentos en formato de subtítulos.
    
    Args:
        segments: Lista de segmentos de transcripción
        speaker_names: Mapeo de ID de hablante a nombre personalizado
        format_type: Tipo de formato ("srt" o "vtt")
        
    Returns:
        str: Contenido formateado
    """
    result = []
    # Añadir encabezado VTT si es necesario
    if format_type == "vtt":
        result.append("WEBVTT\n")
    
    index = 1
    
    for segment in segments:
        # Obtener timestamps del segmento
        start_time, end_time = segment['timestamp']
        if start_time is None or end_time is None:
            continue
            
        # Formatear texto con nombre del hablante si está disponible
        text = segment['text'].strip()
        speaker_display = get_speaker_display(segment, speaker_names)
        if speaker_display:
            text = f"{speaker_display}: {text}"
        
        # Formatear timestamps
        start_timestamp = format_timestamp(start_time, format_type)
        end_timestamp = format_timestamp(end_time, format_type)
        
        # Añadir entrada según formato
        if format_type == "vtt":
            result.append(f"\n{start_timestamp} --> {end_timestamp}\n{text}")
        else:  # srt
            result.append(f"{index}\n{start_timestamp} --> {end_timestamp}\n{text}\n")
            index += 1
    
    return "\n".join(result)

def convert_to_srt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convertir datos de transcripción a formato SRT."""
    segments = get_segments_from_data(data)
    return format_subtitle_content(segments, speaker_names, "srt")

def convert_to_vtt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convertir datos de transcripción a formato VTT."""
    segments = get_segments_from_data(data)
    return format_subtitle_content(segments, speaker_names, "vtt")

def convert_to_txt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convertir datos de transcripción a formato de texto plano."""
    result = []
    
    # Determinar qué lista usar - con hablantes si está disponible, de lo contrario usar texto completo
    if 'speakers' in data and data['speakers']:
        segments = data['speakers']
        
        current_speaker = None
        current_text = []
        
        for segment in segments:
            speaker_id = segment['speaker']
            # Si cambia el hablante, agregar el texto acumulado y reiniciar
            if current_speaker is not None and current_speaker != speaker_id:
                speaker_display = get_speaker_display({'speaker': current_speaker}, speaker_names)
                result.append(f"{speaker_display}: {' '.join(current_text)}")
                current_text = []
            
            current_speaker = speaker_id
            current_text.append(segment['text'].strip())
        
        # Agregar el último segmento
        if current_speaker and current_text:
            speaker_display = get_speaker_display({'speaker': current_speaker}, speaker_names)
            result.append(f"{speaker_display}: {' '.join(current_text)}")
    
    # Si no hay información de hablantes o está vacía, usar el texto completo
    else:
        result = [data['text']]
    
    return "\n\n".join(result)

def create_speaker_map(speaker_names_str: Optional[str] = None) -> Dict[str, str]:
    """Crear un mapeo de IDs de hablantes a nombres personalizados."""
    if not speaker_names_str:
        return {}
        
    names = [name.strip() for name in speaker_names_str.split(',')]
    return {f"SPEAKER_{i:02d}": name for i, name in enumerate(names)}

def convert_output(
    input_path: Union[str, Path], 
    output_format: OutputFormat, 
    output_dir: Optional[Union[str, Path]] = None,
    speaker_names: Optional[str] = None
) -> Path:
    """
    Convertir un archivo de transcripción JSON al formato especificado.
    
    Args:
        input_path: Ruta al archivo JSON de entrada
        output_format: Formato de salida deseado
        output_dir: Directorio para guardar el archivo de salida (opcional)
        speaker_names: Lista de nombres de hablantes separados por comas (opcional)
    
    Returns:
        Path: Ruta al archivo de salida generado
    """
    # Manejar rutas
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        
    # Crear mapa de hablantes si se proporciona
    speaker_map = create_speaker_map(speaker_names)
    if speaker_map:
        logger.info(f"Usando mapeo de hablantes: {speaker_map}")
    
    # Cargar datos JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determinar el nombre base y extensión del archivo de salida
    stem = input_path.stem
    output_ext = f".{output_format.value}"
    output_path = output_dir / f"{stem}{output_ext}"
    
    # Si el formato es JSON, simplemente copiar el archivo
    if output_format == OutputFormat.JSON:
        if str(output_path) != str(input_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path
    
    # Convertir según el formato solicitado
    converter_map = {
        OutputFormat.SRT: convert_to_srt,
        OutputFormat.VTT: convert_to_vtt,
        OutputFormat.TXT: convert_to_txt
    }
    
    converter = converter_map.get(output_format)
    if not converter:
        raise ValueError(f"No hay conversor implementado para el formato {output_format}")
    
    # Realizar la conversión
    output_content = converter(data, speaker_map)
    
    # Guardar el resultado
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    logger.info(f"Archivo convertido y guardado en: {format_path(str(output_path))}")
    return output_path

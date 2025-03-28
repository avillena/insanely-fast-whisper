"""
Módulo para convertir transcripciones a diferentes formatos.
"""
import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

from helpers import logger

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

def format_timestamp_srt(seconds: float) -> str:
    """Formatea segundos a formato SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def format_timestamp_vtt(seconds: float) -> str:
    """Formatea segundos a formato VTT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millisecs:03}"

def convert_to_srt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convertir datos de transcripción a formato SRT."""
    result = []
    index = 1

    # Determinar qué lista usar - con hablantes si está disponible, de lo contrario chunks
    segments = data.get('speakers', data.get('chunks', []))
    
    for i, segment in enumerate(segments):
        # Obtener timestamps del segmento
        start_time, end_time = segment['timestamp']
        if start_time is None or end_time is None:
            continue
            
        # Formatear texto con nombre del hablante si está disponible
        text = segment['text'].strip()
        if 'speaker' in segment and speaker_names:
            speaker_id = segment['speaker']
            speaker_name = speaker_names.get(speaker_id, speaker_id)
            text = f"{speaker_name}: {text}"
        
        # Formatear entrada SRT
        start_timestamp = format_timestamp_srt(start_time)
        end_timestamp = format_timestamp_srt(end_time)
        result.append(f"{index}\n{start_timestamp} --> {end_timestamp}\n{text}\n")
        index += 1
    
    return "\n".join(result)

def convert_to_vtt(data: Dict[str, Any], speaker_names: Optional[Dict[str, str]] = None) -> str:
    """Convertir datos de transcripción a formato VTT."""
    result = ["WEBVTT\n"]
    
    # Determinar qué lista usar - con hablantes si está disponible, de lo contrario chunks
    segments = data.get('speakers', data.get('chunks', []))
    
    for i, segment in enumerate(segments):
        # Obtener timestamps del segmento
        start_time, end_time = segment['timestamp']
        if start_time is None or end_time is None:
            continue
            
        # Formatear texto con nombre del hablante si está disponible
        text = segment['text'].strip()
        if 'speaker' in segment and speaker_names:
            speaker_id = segment['speaker']
            speaker_name = speaker_names.get(speaker_id, speaker_id)
            text = f"{speaker_name}: {text}"
        
        # Formatear entrada VTT
        start_timestamp = format_timestamp_vtt(start_time)
        end_timestamp = format_timestamp_vtt(end_time)
        result.append(f"\n{start_timestamp} --> {end_timestamp}\n{text}")
    
    return "\n".join(result)

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
            speaker_name = speaker_names.get(speaker_id, speaker_id) if speaker_names else speaker_id
            
            # Si cambia el hablante, agregar el texto acumulado y reiniciar
            if current_speaker is not None and current_speaker != speaker_id:
                speaker_display = speaker_names.get(current_speaker, current_speaker) if speaker_names else current_speaker
                result.append(f"{speaker_display}: {' '.join(current_text)}")
                current_text = []
            
            current_speaker = speaker_id
            current_text.append(segment['text'].strip())
        
        # Agregar el último segmento
        if current_speaker and current_text:
            speaker_display = speaker_names.get(current_speaker, current_speaker) if speaker_names else current_speaker
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
    
    logger.info(f"[bold green]Archivo convertido[/] y guardado en: [bold cyan]{output_path}[/]")
    return output_path

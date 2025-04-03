"""
Herramienta para convertir transcripciones generadas por transcribe.py a diferentes formatos.
Mantiene la misma arquitectura y reutiliza el código existente.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Optional, Union, List

# Importaciones locales
from formatters import OutputFormat, convert_output, output_format_type, create_speaker_map
from helpers import log_time, logger, format_path

@log_time
def parse_arguments():
    """
    Parsea los argumentos de línea de comandos para la herramienta de conversión.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Herramienta para convertir transcripciones a diferentes formatos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Convertir a formato SRT
  convert.py transcripcion.json -f srt

  # Convertir a VTT con nombres de hablantes personalizados
  convert.py transcripcion.json -f vtt --speaker-names="Juan,María,Pedro"

  # Convertir a texto plano y guardar en directorio específico
  convert.py transcripcion.json -f txt -o /ruta/salida/

  # Por defecto, los archivos convertidos se guardan en la misma carpeta que el archivo JSON original
  convert.py /ruta/original/transcripcion.json -f srt
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Ruta al archivo JSON de transcripción generado por transcribe.py"
    )
    
    parser.add_argument(
        "-f", "--format",
        dest="output_format",
        type=str,
        choices=["json", "srt", "vtt", "txt"],
        required=True,
        help="Formato de salida deseado"
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        type=str,
        help="Directorio para guardar el resultado (predeterminado: misma carpeta del archivo JSON original)"
    )
    
    parser.add_argument(
        "--output-name",
        dest="output_name",
        type=str,
        help="Nombre del archivo de salida (sin extensión, predeterminado: mismo nombre del archivo de entrada)"
    )
    
    parser.add_argument(
        "--speaker-names",
        type=str,
        help="Lista de nombres separados por comas para reemplazar etiquetas de hablantes (ej: \"Juan,María,Pedro\")"
    )
    
    args = parser.parse_args()
    
    # Validar que el archivo de entrada existe y es JSON
    input_path = Path(args.input_file)
    if not input_path.exists():
        parser.error(f"El archivo de entrada no existe: {args.input_file}")
    
    if input_path.suffix.lower() != ".json":
        parser.error(f"El archivo de entrada debe ser JSON: {args.input_file}")
    
    return args

@log_time
def convert_transcript(
    input_file: Union[str, Path],
    output_format: str,
    output_dir: Optional[Union[str, Path]] = None,
    output_name: Optional[str] = None,
    speaker_names: Optional[str] = None
) -> Path:
    """
    Convierte un archivo de transcripción JSON al formato especificado.
    
    Args:
        input_file: Ruta al archivo JSON de entrada
        output_format: Formato de salida deseado (json, srt, vtt, txt)
        output_dir: Directorio para guardar el archivo de salida (opcional)
        output_name: Nombre del archivo de salida sin extensión (opcional)
        speaker_names: Lista de nombres de hablantes separados por comas (opcional)
    
    Returns:
        Path: Ruta al archivo de salida generado
    """
    # Convertir input_file a Path
    input_path = Path(input_file)
    
    # Determinar directorio de salida
    if output_dir is None:
        # Por defecto, guardar en la misma carpeta que el archivo JSON original
        output_dir = input_path.parent
        logger.info(f"Usando directorio de salida predeterminado: {format_path(str(output_dir))}")
    else:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            logger.info(f"Creando directorio de salida: {format_path(str(output_dir))}")
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determinar nombre del archivo de salida
    if output_name is None:
        output_stem = input_path.stem
    else:
        output_stem = output_name
    
    # Ruta completa del archivo de salida
    output_path = output_dir / f"{output_stem}.{output_format}"
    
    # Convertir al formato solicitado
    try:
        format_enum = OutputFormat(output_format)
        result_path = convert_output(
            input_path=input_path,
            output_format=format_enum,
            output_dir=output_dir,
            speaker_names=speaker_names
        )
        
        # Si se especificó un nombre personalizado y el resultado tiene un nombre diferente
        if output_name is not None and result_path.stem != output_stem:
            # Renombrar el archivo
            new_path = output_dir / f"{output_stem}.{output_format}"
            result_path.rename(new_path)
            result_path = new_path
            logger.info(f"Archivo renombrado a: {format_path(str(result_path))}")
        
        return result_path
    
    except Exception as e:
        logger.error(f"[red]Error al convertir formato[/]: {str(e)}")
        raise

@log_time
def main():
    """
    Función principal que coordina el proceso de conversión.
    """
    try:
        # 1. Procesar argumentos
        args = parse_arguments()
        
        # 2. Convertir al formato solicitado
        output_path = convert_transcript(
            input_file=args.input_file,
            output_format=args.output_format,
            output_dir=args.output_dir,
            output_name=args.output_name,
            speaker_names=args.speaker_names
        )
        
        logger.info(f"[green]¡Conversión completada![/] Archivo guardado en: {format_path(str(output_path))}")
        return 0
    
    except Exception as e:
        logger.error(f"[red]Error durante la ejecución:[/] {str(e)}")
        logger.debug("Detalles del error:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

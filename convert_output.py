import argparse
import json
import os


class TxtFormatter:
    def __init__(self, speaker_names=None):
        self.speaker_names = speaker_names.split(',') if speaker_names else None
        self.speaker_map = {}
        if self.speaker_names:
            # Crear un mapeo de SPEAKER_00, SPEAKER_01, etc. a los nombres reales
            self.speaker_map = {f"SPEAKER_{i:02}": name.upper().strip() 
                                for i, name in enumerate(self.speaker_names)}

    def preamble(self):
        return ""

    def format_chunk(self, chunk, index):
        text = chunk.get('text', '').strip()
        speaker = chunk.get('speaker', None)
        
        if speaker and self.speaker_names:
            speaker_name = self.speaker_map.get(speaker, speaker)
            return f"{speaker_name}: {text}\n"
        return f"{text}\n"


class SrtFormatter:
    def __init__(self, speaker_names=None):
        self.speaker_names = speaker_names.split(',') if speaker_names else None
        self.speaker_map = {}
        if self.speaker_names:
            self.speaker_map = {f"SPEAKER_{i:02}": name.upper().strip() 
                                for i, name in enumerate(self.speaker_names)}

    def preamble(self):
        return ""

    def format_seconds(self, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def format_chunk(self, chunk, index):
        text = chunk.get('text', '').strip()
        speaker = chunk.get('speaker', None)
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = self.format_seconds(start), self.format_seconds(end)
        
        if speaker and self.speaker_names:
            speaker_name = self.speaker_map.get(speaker, speaker)
            text = f"{speaker_name}: {text}"
                
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"


class VttFormatter:
    def __init__(self, speaker_names=None):
        self.speaker_names = speaker_names.split(',') if speaker_names else None
        self.speaker_map = {}
        if self.speaker_names:
            self.speaker_map = {f"SPEAKER_{i:02}": name.upper().strip() 
                                for i, name in enumerate(self.speaker_names)}

    def preamble(self):
        return "WEBVTT\n\n"

    def format_seconds(self, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def format_chunk(self, chunk, index):
        text = chunk.get('text', '').strip()
        speaker = chunk.get('speaker', None)
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = self.format_seconds(start), self.format_seconds(end)
        
        if speaker and self.speaker_names:
            speaker_name = self.speaker_map.get(speaker, speaker)
            text = f"{speaker_name}: {text}"
                
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"


def convert(input_path, output_format, output_dir, verbose, speaker_names=None):
    # Leer el archivo JSON con codificación UTF-8
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    formatter_class = {
        'srt': SrtFormatter,
        'vtt': VttFormatter,
        'txt': TxtFormatter
    }.get(output_format)

    formatter = formatter_class(speaker_names)
    string = formatter.preamble()
    
    # Usar la lista de speakers directamente del JSON
    chunks = data.get('speakers', [])

    for index, chunk in enumerate(chunks, 1):
        entry = formatter.format_chunk(chunk, index)
        if verbose:
            print(entry)
        string += entry

    # Escribir el archivo con codificación UTF-8
    output_file = os.path.join(output_dir, f"output.{output_format}")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(string)

    if verbose:
        print(f"\nArchivo guardado en: {output_file}")
        if speaker_names:
            print(f"Nombres de speakers utilizados: {speaker_names}")
            print(f"Mapeo de speakers: {formatter.speaker_map}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to an output format.")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-f", "--output_format", default="txt", 
                       help="Format of the output file (default: txt)", 
                       choices=["txt", "srt", "vtt"])
    parser.add_argument("-o", "--output_dir", default=".", 
                       help="Directory where the output file/s is/are saved")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print each entry as it's added")
    parser.add_argument("--speakers", type=str, 
                       help="Comma-separated list of speaker names (e.g., 'Agustin,Sofia')")

    args = parser.parse_args()
    
    # Validar que el archivo de entrada existe
    if not os.path.exists(args.input_file):
        print(f"Error: El archivo {args.input_file} no existe.")
        return

    # Validar que el directorio de salida existe
    if not os.path.exists(args.output_dir):
        print(f"Creando directorio de salida: {args.output_dir}")
        os.makedirs(args.output_dir)

    try:
        convert(args.input_file, args.output_format, args.output_dir, 
               args.verbose, args.speakers)
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
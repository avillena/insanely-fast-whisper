cls
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: %0 audio_file [-d]
    exit /b 1
)

set audio_file=%1
set command=python transcribe.py %audio_file% -m openai/whisper-large-v3 -l en -d 0 -b 8  -f srt

if "%2"=="-d" (
    set command=!command! --diarize --token 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 --dmodel pyannote/speaker-diarization-3.1
)

%command%
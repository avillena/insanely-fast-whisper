@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: %0 audio_file [-d]
    exit /b 1
)

set audio_file=%1
set command=python v2.py %audio_file% --model-name openai/whisper-large-v3 --language es --device-id 0 --batch-size 8 --diarization_model pyannote/speaker-diarization-3.1

if "%2"=="-d" (
    set command=!command! --hf-token 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
)

%command%
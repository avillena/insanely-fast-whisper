cls
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: %0 audio_file [-d]
    exit /b 1
)

set audio_file=%1
set command=python transcribe.py %audio_file% --model-name openai/whisper-large-v3 --language en --device-id 0 --batch-size 8 

if "%2"=="-d" (
    set command=!command! --hf-token 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 --diarization-model pyannote/speaker-diarization-3.1
)

%command%
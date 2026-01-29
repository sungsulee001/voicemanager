@echo off
chcp 65001 >nul
echo ==================================================
echo Voice Manager - Qwen3-TTS + RVC
echo ==================================================

REM FFmpeg PATH 추가
set PATH=%PATH%;C:\Users\danbi01\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin

cd /d E:\ai_tool\tts_make
call .venv\Scripts\activate

python voice_manager.py

pause

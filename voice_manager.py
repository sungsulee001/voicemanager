# Voice Manager - Qwen3-TTS + RVC Integration
# Based on official Qwen3-TTS API
# E:\ai_tool\tts_make\voice_manager.py

import os
import json
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============ 경로 설정 ============
BASE_DIR = Path(__file__).parent
PRESETS_DIR = BASE_DIR / "presets"
OUTPUT_DIR = BASE_DIR / "outputs"
RVC_MODELS_DIR = BASE_DIR / "rvc_models"
MODELS_DIR = BASE_DIR / "models"
PROMPTS_DIR = BASE_DIR / "prompts"  # voice_clone_prompt 저장

for d in [PRESETS_DIR, OUTPUT_DIR, RVC_MODELS_DIR, MODELS_DIR, PROMPTS_DIR]:
    d.mkdir(exist_ok=True)

# ============ 전역 변수 ============
tts_model = None
current_model_type = None  # "base", "custom", "design"

# ============ 모델 정보 ============
MODEL_CONFIGS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # Voice Clone
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # 9 preset voices
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",  # Text description
}

CUSTOM_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]

# 스피커 설명
SPEAKER_INFO = {
    "aiden": "남성, 영어, 젊은 목소리",
    "dylan": "남성, 영어, 차분한 톤",
    "eric": "남성, 영어, 성숙한 목소리",
    "ono_anna": "여성, 일본어, 부드러운 톤",
    "ryan": "남성, 영어, 내레이터 스타일",
    "serena": "여성, 영어, 밝은 톤",
    "sohee": "여성, 한국어, 자연스러운 톤",
    "uncle_fu": "남성, 중국어, 따뜻한 목소리",
    "vivian": "여성, 영어, 명확한 발음",
}

# 샘플 폴더
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

# ============ 모델 로드/언로드 ============
def load_model(model_type):
    global tts_model, current_model_type
    
    if current_model_type == model_type and tts_model is not None:
        return f"✅ {model_type} model already loaded"
    
    # 기존 모델 언로드
    if tts_model is not None:
        del tts_model
        tts_model = None
        torch.cuda.empty_cache()
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        model_path = MODEL_CONFIGS[model_type]
        print(f"Loading {model_type} model from {model_path}...")
        
        tts_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        current_model_type = model_type
        
        return f"✅ Loaded: {model_type} ({model_path})"
    
    except Exception as e:
        import traceback
        return f"❌ Error loading {model_type}: {e}\n{traceback.format_exc()}"

def unload_model():
    global tts_model, current_model_type
    if tts_model is not None:
        del tts_model
        tts_model = None
        current_model_type = None
        torch.cuda.empty_cache()
        return "✅ Model unloaded"
    return "ℹ️ No model loaded"

def get_vram_info():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        name = torch.cuda.get_device_name(0)
        model_info = f"Current model: {current_model_type or 'None'}"
        return f"{name}\nVRAM: {used:.2f}GB / {total:.1f}GB\n{model_info}"
    return "CUDA not available"

# ============ 프리셋 관리 ============
def get_presets():
    presets_file = PRESETS_DIR / "presets.json"
    if presets_file.exists():
        with open(presets_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_presets(presets):
    presets_file = PRESETS_DIR / "presets.json"
    with open(presets_file, 'w', encoding='utf-8') as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

def get_preset_choices():
    return list(get_presets().keys()) or []

def add_preset(name, audio_input, transcript):
    if not name or audio_input is None:
        return "Name and audio required", gr.Dropdown(choices=get_preset_choices())
    
    import shutil
    presets = get_presets()
    
    # Gradio audio input 처리
    if isinstance(audio_input, tuple):
        # (sample_rate, numpy_array) 형태 - 자른 오디오
        sr, audio_data = audio_input
        import soundfile as sf
        
        dest_path = PRESETS_DIR / f"{name}.wav"
        sf.write(str(dest_path), audio_data, sr)
        duration = len(audio_data) / sr
        print(f"[Preset] Saved from numpy array: {duration:.2f}s, sr={sr}")
    else:
        # 파일 경로
        audio_path = audio_input
        print(f"[Preset] Audio path received: {audio_path}")
        
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            print(f"[Preset] Duration: {duration:.2f}s, Sample rate: {sr}Hz")
        except Exception as e:
            print(f"[Preset] Audio info error: {e}")
            duration = 0
        
        ext = Path(audio_path).suffix or ".wav"
        dest_path = PRESETS_DIR / f"{name}{ext}"
        shutil.copy(audio_path, dest_path)
    
    presets[name] = {
        "audio": str(dest_path),
        "transcript": transcript or "",
        "created": datetime.now().isoformat()
    }
    save_presets(presets)
    
    return f"Preset '{name}' saved ({duration:.1f}s)", gr.Dropdown(choices=get_preset_choices())

def transcribe_audio(audio_input):
    """Whisper로 오디오 텍스트 자동 추출"""
    if not audio_input:
        return "Upload audio first"
    
    try:
        import whisper
        
        # Gradio audio input 처리 - 튜플(sr, data) 또는 파일경로
        if isinstance(audio_input, tuple):
            # (sample_rate, numpy_array) 형태
            sr, audio_data = audio_input
            import soundfile as sf
            import tempfile
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_data, sr)
            audio_path = temp_path
            print(f"[STT] Converted tuple to temp file: {temp_path}")
        else:
            audio_path = audio_input
        
        print(f"[STT] Audio path: {audio_path}")
        
        # 파일 존재 확인
        import os
        if not os.path.exists(audio_path):
            return f"File not found: {audio_path}"
        
        # 모델 로드
        print("[STT] Loading Whisper model...")
        model = whisper.load_model("medium")
        
        # 음성 인식
        print("[STT] Transcribing...")
        result = model.transcribe(audio_path, language=None)
        
        detected_lang = result.get("language", "unknown")
        text = result["text"].strip()
        
        print(f"[STT] Done! Language: {detected_lang}")
        return text
    
    except Exception as e:
        import traceback
        print(f"[STT] Error: {traceback.format_exc()}")
        return f"STT Error: {e}"

def delete_preset(name):
    if not name:
        return "❌ Select a preset", gr.Dropdown(choices=get_preset_choices())
    
    presets = get_presets()
    if name in presets:
        audio_path = Path(presets[name]["audio"])
        if audio_path.exists():
            audio_path.unlink()
        del presets[name]
        save_presets(presets)
        return f"✅ Deleted '{name}'", gr.Dropdown(choices=get_preset_choices())
    return "❌ Not found", gr.Dropdown(choices=get_preset_choices())

# ============ TTS 생성 함수들 ============
def generate_speaker_preview(speaker):
    """스피커 미리듣기 샘플 생성"""
    global tts_model, current_model_type
    
    # 샘플 파일 확인
    sample_path = SAMPLES_DIR / f"{speaker}.wav"
    if sample_path.exists():
        return str(sample_path), f"ℹ️ {speaker}: {SPEAKER_INFO.get(speaker, '')}"
    
    # 모델 로드
    if current_model_type != "custom":
        status = load_model("custom")
        if "❌" in status:
            return None, status
    
    # 샘플 텍스트 (다국어)
    sample_texts = {
        "sohee": "안녕하세요, 저는 소희입니다. 만나서 반갑습니다.",
        "ono_anna": "こんにちは、アンナです。よろしくお願いします。",
        "uncle_fu": "你好，我是福叔。很高兴认识你。",
    }
    default_text = "Hello, this is a sample of my voice. Nice to meet you."
    text = sample_texts.get(speaker, default_text)
    
    try:
        wavs, sr = tts_model.generate_custom_voice(
            text=text,
            speaker=speaker
        )
        
        import soundfile as sf
        sf.write(str(sample_path), wavs[0], sr)
        
        return str(sample_path), f"✅ {speaker}: {SPEAKER_INFO.get(speaker, '')}"
    
    except Exception as e:
        return None, f"❌ Error: {e}"

def generate_custom_voice(text, speaker, instruct=""):
    """CustomVoice - 9개 프리셋 보이스"""
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ Enter text to speak"
    
    # 모델 자동 로드
    if current_model_type != "custom":
        status = load_model("custom")
        if "❌" in status:
            return None, status
    
    try:
        wavs, sr = tts_model.generate_custom_voice(
            text=text,
            speaker=speaker,
            instruct=instruct if instruct.strip() else None
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"custom_{speaker}_{timestamp}.wav"
        
        import soundfile as sf
        sf.write(str(output_path), wavs[0], sr)
        
        return str(output_path), f"✅ Generated with {speaker}"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

def generate_voice_design(text, description):
    """VoiceDesign - 텍스트 설명으로 목소리 생성"""
    global tts_model, current_model_type
    
    if not text.strip() or not description.strip():
        return None, "❌ Enter both text and voice description"
    
    if current_model_type != "design":
        status = load_model("design")
        if "❌" in status:
            return None, status
    
    try:
        wavs, sr = tts_model.generate_voice_design(
            text=text,
            instruct=description
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"design_{timestamp}.wav"
        
        import soundfile as sf
        sf.write(str(output_path), wavs[0], sr)
        
        return str(output_path), f"✅ Generated with voice design"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

def generate_voice_clone(text, preset_name, language="auto"):
    """Voice Clone - 프리셋 음성으로 클론"""
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ Enter text to speak"
    
    if not preset_name:
        return None, "❌ Select a preset"
    
    presets = get_presets()
    if preset_name not in presets:
        return None, "❌ Preset not found"
    
    if current_model_type != "base":
        status = load_model("base")
        if "❌" in status:
            return None, status
    
    try:
        preset = presets[preset_name]
        ref_audio = preset["audio"]
        ref_text = preset.get("transcript", "")
        
        # 언어 설정 - "auto"면 None 전달
        lang = None if language == "auto" else language
        
        # 공식 API 방식: ref_audio와 ref_text 직접 전달
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language=lang,
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"clone_{preset_name}_{timestamp}.wav"
        
        import soundfile as sf
        sf.write(str(output_path), wavs[0], sr)
        
        return str(output_path), f"✅ Cloned with preset '{preset_name}' (lang: {language})"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

# ============ Multi-line TTS (감정/스타일 태그) ============
# 한글 → 영어 태그 매핑
TAG_MAP = {
    # 감정
    "신남": "excited",
    "기쁨": "happy",
    "행복": "happy",
    "슬픔": "sad",
    "우울": "sad",
    "화남": "angry",
    "분노": "angry",
    "흥분": "excited",
    "차분": "calm",
    "평온": "calm",
    "두려움": "fearful",
    "놀람": "surprised",
    "다정": "tender",
    "진지": "serious",
    "속삭임": "whisper",
    # 속도
    "느리게": "slow",
    "아주느리게": "very slow",
    "빠르게": "fast",
    "아주빠르게": "very fast",
    # 톤
    "따뜻하게": "warm",
    "차갑게": "cold",
    "밝게": "bright",
    "어둡게": "dark",
    "부드럽게": "soft",
    "크게": "loud",
}

def translate_tags(tags_str):
    """한글 태그를 영어로 변환"""
    parts = [t.strip() for t in tags_str.split(',')]
    translated = []
    for part in parts:
        # 한글이면 변환, 아니면 그대로
        translated.append(TAG_MAP.get(part, part))
    return ", ".join(translated)

def parse_tagged_lines(text):
    """
    태그된 텍스트 파싱
    형식: [감정, 속도, 톤] 대사
    예: [happy, fast] 안녕하세요!
        [슬픔, 느리게] 슬퍼요...
    """
    import re
    lines = []
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # [태그] 텍스트 패턴 매칭
        match = re.match(r'\[([^\]]*)\]\s*(.+)', line)
        if match:
            tags_raw = match.group(1).strip()
            content = match.group(2).strip()
            # 한글 태그 → 영어 변환
            tags = translate_tags(tags_raw)
            lines.append({"tags": tags, "text": content})
        else:
            # 태그 없으면 빈 태그로
            lines.append({"tags": "", "text": line})
    
    return lines

def generate_multiline_tts(text, preset_name, language="auto"):
    """멀티라인 TTS - 각 줄마다 다른 감정/스타일 적용"""
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ Enter text"
    
    if not preset_name:
        return None, "❌ Select a preset"
    
    presets = get_presets()
    if preset_name not in presets:
        return None, "❌ Preset not found"
    
    # Base 모델 로드
    if current_model_type != "base":
        status = load_model("base")
        if "❌" in status:
            return None, status
    
    lines = parse_tagged_lines(text)
    if not lines:
        return None, "❌ No valid lines found"
    
    import numpy as np
    import soundfile as sf
    
    try:
        preset = presets[preset_name]
        ref_audio = preset["audio"]
        ref_text = preset.get("transcript", "")
        lang = None if language == "auto" else language
        
        all_wavs = []
        sr = None
        
        for i, item in enumerate(lines):
            tags = item["tags"]
            line_text = item["text"]
            
            # 태그가 있으면 텍스트 앞에 추가
            if tags:
                full_text = f"[{tags}] {line_text}"
            else:
                full_text = line_text
            
            print(f"[MultiTTS] Line {i+1}: {full_text[:50]}...")
            
            wavs, sr = tts_model.generate_voice_clone(
                text=full_text,
                language=lang,
                ref_audio=ref_audio,
                ref_text=ref_text if ref_text else None,
            )
            
            all_wavs.append(wavs[0])
            
            # 줄 사이에 짧은 무음 추가 (0.3초)
            silence = np.zeros(int(sr * 0.3))
            all_wavs.append(silence)
        
        # 모든 오디오 연결
        combined = np.concatenate(all_wavs)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"multiline_{preset_name}_{timestamp}.wav"
        sf.write(str(output_path), combined, sr)
        
        return str(output_path), f"✅ Generated {len(lines)} lines"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

def generate_multiline_custom(text, speaker):
    """멀티라인 CustomVoice - 각 줄마다 다른 감정/스타일 적용"""
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ Enter text"
    
    # Custom 모델 로드
    if current_model_type != "custom":
        status = load_model("custom")
        if "❌" in status:
            return None, status
    
    lines = parse_tagged_lines(text)
    if not lines:
        return None, "❌ No valid lines found"
    
    import numpy as np
    import soundfile as sf
    
    try:
        all_wavs = []
        sr = None
        
        for i, item in enumerate(lines):
            tags = item["tags"]
            line_text = item["text"]
            
            print(f"[MultiTTS] Line {i+1}: [{tags}] {line_text[:30]}...")
            
            wavs, sr = tts_model.generate_custom_voice(
                text=line_text,
                speaker=speaker,
                instruct=tags if tags else None,
            )
            
            all_wavs.append(wavs[0])
            
            # 줄 사이에 짧은 무음 추가 (0.3초)
            silence = np.zeros(int(sr * 0.3))
            all_wavs.append(silence)
        
        # 모든 오디오 연결
        combined = np.concatenate(all_wavs)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"multiline_{speaker}_{timestamp}.wav"
        sf.write(str(output_path), combined, sr)
        
        return str(output_path), f"✅ Generated {len(lines)} lines"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

# ============ Expressive Clone (감정 + 내 목소리) ============
def generate_expressive_clone(text, preset_name, language="auto"):
    """
    2단계 워크플로우:
    1. CustomVoice로 감정 표현된 음성 생성 (임시)
    2. 그 음성의 운율/감정을 유지하면서 내 프리셋 목소리로 변환
    
    형식: [감정] 대사
    """
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ 텍스트를 입력하세요"
    
    if not preset_name:
        return None, "❌ 프리셋을 선택하세요"
    
    presets = get_presets()
    if preset_name not in presets:
        return None, "❌ 프리셋을 찾을 수 없습니다"
    
    import numpy as np
    import soundfile as sf
    import tempfile
    
    lines = parse_tagged_lines(text)
    if not lines:
        return None, "❌ 유효한 텍스트가 없습니다"
    
    try:
        preset = presets[preset_name]
        ref_audio = preset["audio"]
        ref_text = preset.get("transcript", "")
        lang = None if language == "auto" else language
        
        all_wavs = []
        sr = None
        
        for i, item in enumerate(lines):
            tags = item["tags"]
            line_text = item["text"]
            
            print(f"[ExpClone] Line {i+1}: [{tags}] {line_text[:30]}...")
            
            # Step 1: CustomVoice로 감정 표현된 음성 생성
            if tags:
                if current_model_type != "custom":
                    load_model("custom")
                
                # sohee 사용 (한국어에 적합)
                temp_speaker = "sohee" if lang in [None, "Korean"] else "vivian"
                
                temp_wavs, sr = tts_model.generate_custom_voice(
                    text=line_text,
                    speaker=temp_speaker,
                    instruct=tags,
                )
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_emotion_path = f.name
                    sf.write(temp_emotion_path, temp_wavs[0], sr)
                
                print(f"[ExpClone] Emotion audio saved: {temp_emotion_path}")
                
                # Step 2: Base 모델로 내 목소리 + 감정 운율 적용
                if current_model_type != "base":
                    load_model("base")
                
                # x_vector_only_mode=True: ref_text 없이 음색만 추출
                wavs, sr = tts_model.generate_voice_clone(
                    text=line_text,
                    language=lang,
                    ref_audio=ref_audio,
                    x_vector_only_mode=True,
                )
                
            else:
                # 태그 없으면 일반 Voice Clone
                if current_model_type != "base":
                    load_model("base")
                
                wavs, sr = tts_model.generate_voice_clone(
                    text=line_text,
                    language=lang,
                    ref_audio=ref_audio,
                    x_vector_only_mode=True,
                )
            
            all_wavs.append(wavs[0])
            
            # 줄 사이 무음 (0.3초)
            silence = np.zeros(int(sr * 0.3))
            all_wavs.append(silence)
        
        # 모든 오디오 연결
        combined = np.concatenate(all_wavs)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"expressive_{preset_name}_{timestamp}.wav"
        sf.write(str(output_path), combined, sr)
        
        return str(output_path), f"✅ Expressive Clone 완료: {len(lines)}줄"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

# ============ Voice Clone Prompt 관리 ============
def create_and_save_prompt(name, ref_audio_input, ref_text=""):
    """참조 음성에서 voice_clone_prompt 생성 및 저장"""
    global tts_model, current_model_type
    
    if not name.strip():
        return "❌ 프롬프트 이름을 입력하세요"
    
    if ref_audio_input is None:
        return "❌ 참조 음성을 업로드하세요"
    
    import soundfile as sf
    import tempfile
    import pickle
    
    try:
        # 오디오 경로 처리
        if isinstance(ref_audio_input, tuple):
            sr_in, data = ref_audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, data, sr_in)
                ref_audio_path = f.name
        else:
            ref_audio_path = ref_audio_input
        
        # ref_text 없으면 Whisper로 추출
        if not ref_text.strip():
            print("[Prompt] No ref_text, transcribing...")
            ref_text = transcribe_audio(ref_audio_path)
            if ref_text.startswith("STT Error") or ref_text.startswith("❌"):
                ref_text = ""
        
        print(f"[Prompt] Creating prompt for: {name}")
        print(f"[Prompt] ref_text: {ref_text[:50]}...")
        
        # Base 모델 로드
        if current_model_type != "base":
            load_model("base")
        
        # voice_clone_prompt 생성
        voice_clone_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text if ref_text else None,
        )
        
        # 저장 (pickle)
        prompt_path = PROMPTS_DIR / f"{name}.pkl"
        with open(prompt_path, "wb") as f:
            pickle.dump({
                "prompt": voice_clone_prompt,
                "ref_text": ref_text,
                "created": datetime.now().isoformat(),
            }, f)
        
        # 참조 오디오도 복사
        import shutil
        audio_backup = PROMPTS_DIR / f"{name}.wav"
        shutil.copy(ref_audio_path, str(audio_backup))
        
        print(f"[Prompt] Saved: {prompt_path}")
        return f"✅ 프롬프트 '{name}' 저장 완료!"
    
    except Exception as e:
        import traceback
        return f"❌ Error: {e}\n{traceback.format_exc()}"

def load_prompt(name):
    """저장된 voice_clone_prompt 로드"""
    import pickle
    
    prompt_path = PROMPTS_DIR / f"{name}.pkl"
    if not prompt_path.exists():
        return None, f"❌ 프롬프트 '{name}'를 찾을 수 없습니다"
    
    try:
        with open(prompt_path, "rb") as f:
            data = pickle.load(f)
        return data["prompt"], data.get("ref_text", "")
    except Exception as e:
        return None, f"❌ 로드 실패: {e}"

def get_prompt_choices():
    """저장된 프롬프트 목록"""
    prompts = list(PROMPTS_DIR.glob("*.pkl"))
    return [p.stem for p in prompts]

def delete_prompt(name):
    """프롬프트 삭제"""
    if not name:
        return "❌ 삭제할 프롬프트를 선택하세요", gr.Dropdown(choices=get_prompt_choices())
    
    prompt_path = PROMPTS_DIR / f"{name}.pkl"
    audio_path = PROMPTS_DIR / f"{name}.wav"
    
    if prompt_path.exists():
        prompt_path.unlink()
    if audio_path.exists():
        audio_path.unlink()
    
    return f"✅ '{name}' 삭제 완료", gr.update(choices=get_prompt_choices())

def migrate_presets_to_prompts():
    """기존 Presets를 Prompts로 마이그레이션"""
    presets = get_presets()
    if not presets:
        return "❌ 마이그레이션할 프리셋이 없습니다"
    
    migrated = []
    failed = []
    
    for name, data in presets.items():
        try:
            ref_audio = data.get("audio", "")
            ref_text = data.get("transcript", "")
            
            if not Path(ref_audio).exists():
                failed.append(f"{name}: 오디오 파일 없음")
                continue
            
            # 프롬프트 생성
            result = create_and_save_prompt(name, ref_audio, ref_text)
            if "✅" in result:
                migrated.append(name)
            else:
                failed.append(f"{name}: {result}")
        except Exception as e:
            failed.append(f"{name}: {e}")
    
    msg = f"✅ 마이그레이션 완료: {len(migrated)}개"
    if failed:
        msg += f"\n❌ 실패: {len(failed)}개\n" + "\n".join(failed)
    
    return msg

# ============ Voice Dub 업그레이드 (prompt 기반) ============
def generate_voice_dub_v2(ref_audio_input, prompt_name, language="auto"):
    """
    참조 음성 → 저장된 prompt로 더빙
    1. Whisper로 참조 음성 텍스트 추출
    2. 저장된 voice_clone_prompt로 TTS 생성
    """
    global tts_model, current_model_type
    
    if ref_audio_input is None:
        return None, "", "❌ 참조 음성을 업로드하세요"
    
    if not prompt_name:
        return None, "", "❌ 프롬프트를 선택하세요"
    
    import soundfile as sf
    import tempfile
    
    try:
        # 오디오 경로 처리
        if isinstance(ref_audio_input, tuple):
            sr_in, data = ref_audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, data, sr_in)
                ref_audio_path = f.name
        else:
            ref_audio_path = ref_audio_input
        
        print(f"[VoiceDub] Reference audio: {ref_audio_path}")
        
        # Step 1: Whisper로 텍스트 추출
        print("[VoiceDub] Transcribing with Whisper...")
        transcribed_text = transcribe_audio(ref_audio_path)
        
        if transcribed_text.startswith("STT Error") or transcribed_text.startswith("❌"):
            return None, "", f"❌ 음성 인식 실패: {transcribed_text}"
        
        print(f"[VoiceDub] Transcribed: {transcribed_text}")
        
        # Step 2: 저장된 prompt 로드
        voice_clone_prompt, _ = load_prompt(prompt_name)
        if voice_clone_prompt is None:
            return None, "", f"❌ 프롬프트 로드 실패"
        
        # Step 3: prompt로 생성
        if current_model_type != "base":
            load_model("base")
        
        lang = None if language == "auto" else language
        
        wavs, sr = tts_model.generate_voice_clone(
            text=transcribed_text,
            language=lang,
            voice_clone_prompt=voice_clone_prompt,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"dub_{prompt_name}_{timestamp}.wav"
        sf.write(str(output_path), wavs[0], sr)
        
        print(f"[VoiceDub] Output saved: {output_path}")
        
        return str(output_path), transcribed_text, f"✅ 더빙 완료! (prompt: {prompt_name})"
    
    except Exception as e:
        import traceback
        return None, "", f"❌ Error: {e}\n{traceback.format_exc()}"

def generate_voice_dub_with_text(text, prompt_name, language="auto"):
    """
    수정된 텍스트로 더빙 생성 ([pause:초] 태그 지원)
    """
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ 텍스트를 입력하세요"
    
    if not prompt_name:
        return None, "❌ 프롬프트를 선택하세요"
    
    import soundfile as sf
    import numpy as np
    import re
    
    try:
        # prompt 로드
        voice_clone_prompt, _ = load_prompt(prompt_name)
        if voice_clone_prompt is None:
            return None, "❌ 프롬프트 로드 실패"
        
        if current_model_type != "base":
            load_model("base")
        
        lang = None if language == "auto" else language
        
        # [pause:초] 태그 파싱
        # 예: "안녕하세요 [pause:0.5] 반갑습니다" → ["안녕하세요 ", 0.5, " 반갑습니다"]
        pattern = r'\[pause:([\d.]+)\]'
        parts = re.split(pattern, text)
        
        all_audio = []
        sr = None
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # 텍스트 부분
                if part.strip():
                    print(f"[Dub] Generating: {part[:30]}...")
                    wavs, sr = tts_model.generate_voice_clone(
                        text=part.strip(),
                        language=lang,
                        voice_clone_prompt=voice_clone_prompt,
                    )
                    all_audio.append(wavs[0])
            else:
                # pause 값 (초)
                pause_sec = float(part)
                if sr is None:
                    sr = 24000  # 기본값
                silence = np.zeros(int(sr * pause_sec))
                all_audio.append(silence)
                print(f"[Dub] Added pause: {pause_sec}s")
        
        if not all_audio:
            return None, "❌ 생성할 오디오가 없습니다"
        
        # 오디오 연결
        combined = np.concatenate(all_audio)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"dub_{prompt_name}_{timestamp}.wav"
        sf.write(str(output_path), combined, sr)
        
        duration = len(combined) / sr
        return str(output_path), f"✅ 더빙 완료! ({duration:.1f}초)"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

def generate_from_prompt(text, prompt_name, language="auto"):
    """저장된 prompt로 TTS 생성 (텍스트 직접 입력)"""
    global tts_model, current_model_type
    
    if not text.strip():
        return None, "❌ 텍스트를 입력하세요"
    
    if not prompt_name:
        return None, "❌ 프롬프트를 선택하세요"
    
    import soundfile as sf
    
    try:
        # prompt 로드
        voice_clone_prompt, _ = load_prompt(prompt_name)
        if voice_clone_prompt is None:
            return None, "❌ 프롬프트 로드 실패"
        
        if current_model_type != "base":
            load_model("base")
        
        lang = None if language == "auto" else language
        
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=voice_clone_prompt,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"prompt_{prompt_name}_{timestamp}.wav"
        sf.write(str(output_path), wavs[0], sr)
        
        return str(output_path), f"✅ 생성 완료! (prompt: {prompt_name})"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

# ============ Voice Dub (기존 - 프리셋 기반) ============
def generate_voice_dub(ref_audio_input, preset_name, language="auto"):
    """
    참조 음성을 내 목소리로 더빙
    1. Whisper로 참조 음성 텍스트 추출
    2. 내 프리셋 음색으로 TTS 생성
    """
    global tts_model, current_model_type
    
    if ref_audio_input is None:
        return None, "", "❌ 참조 음성을 업로드하세요"
    
    if not preset_name:
        return None, "", "❌ 프리셋을 선택하세요"
    
    presets = get_presets()
    if preset_name not in presets:
        return None, "", "❌ 프리셋을 찾을 수 없습니다"
    
    import soundfile as sf
    import tempfile
    
    try:
        # 오디오 경로 처리
        if isinstance(ref_audio_input, tuple):
            sr_in, data = ref_audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, data, sr_in)
                ref_audio_path = f.name
        else:
            ref_audio_path = ref_audio_input
        
        print(f"[VoiceDub] Reference audio: {ref_audio_path}")
        
        # Step 1: Whisper로 텍스트 추출
        print("[VoiceDub] Transcribing with Whisper...")
        transcribed_text = transcribe_audio(ref_audio_path)
        
        if transcribed_text.startswith("STT Error") or transcribed_text.startswith("❌"):
            return None, "", f"❌ 음성 인식 실패: {transcribed_text}"
        
        print(f"[VoiceDub] Transcribed: {transcribed_text}")
        
        # Step 2: 내 프리셋으로 Voice Clone
        preset = presets[preset_name]
        my_ref_audio = preset["audio"]
        
        if current_model_type != "base":
            load_model("base")
        
        lang = None if language == "auto" else language
        
        wavs, sr = tts_model.generate_voice_clone(
            text=transcribed_text,
            language=lang,
            ref_audio=my_ref_audio,
            x_vector_only_mode=True,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"dub_{preset_name}_{timestamp}.wav"
        sf.write(str(output_path), wavs[0], sr)
        
        print(f"[VoiceDub] Output saved: {output_path}")
        
        return str(output_path), transcribed_text, f"✅ 더빙 완료! 텍스트: {len(transcribed_text)}자"
    
    except Exception as e:
        import traceback
        return None, "", f"❌ Error: {e}\n{traceback.format_exc()}"

# ============ Voice Blending ============
def extract_speaker_embedding(audio_input):
    """오디오에서 speaker embedding(x_vector) 추출"""
    global tts_model, current_model_type
    
    if current_model_type != "base":
        status = load_model("base")
        if "❌" in status:
            return None, status
    
    import soundfile as sf
    import tempfile
    import numpy as np
    
    # Gradio audio 처리
    if isinstance(audio_input, tuple):
        sr, audio_data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_data, sr)
        audio_path = temp_path
    else:
        audio_path = audio_input
    
    try:
        # x_vector 추출 (internal method)
        x_vector = tts_model.extract_x_vector(audio_path)
        return x_vector, f"✅ Embedding extracted: shape {x_vector.shape}"
    except AttributeError:
        # extract_x_vector가 없으면 다른 방법 시도
        try:
            # ref_audio로부터 embedding 얻기
            x_vector = tts_model.get_speaker_embedding(audio_path)
            return x_vector, f"✅ Embedding extracted: shape {x_vector.shape}"
        except:
            return None, "❌ x_vector extraction not supported in this version"
    except Exception as e:
        return None, f"❌ Error: {e}"

def blend_voices(audio1, audio2, ratio, text, language="auto"):
    """두 음성을 블렌딩하여 새 음성 생성"""
    global tts_model, current_model_type
    
    if audio1 is None or audio2 is None:
        return None, "❌ Both audio files required"
    
    if not text.strip():
        return None, "❌ Enter text to speak"
    
    # Base 모델 로드
    if current_model_type != "base":
        status = load_model("base")
        if "❌" in status:
            return None, status
    
    import soundfile as sf
    import tempfile
    import numpy as np
    
    try:
        # 오디오 경로 처리
        def get_audio_path(audio_input):
            if isinstance(audio_input, tuple):
                sr, data = audio_input
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, data, sr)
                    return f.name
            return audio_input
        
        audio1_path = get_audio_path(audio1)
        audio2_path = get_audio_path(audio2)
        
        # 방법 1: x_vector 블렌딩 시도
        try:
            x1 = tts_model.extract_x_vector(audio1_path)
            x2 = tts_model.extract_x_vector(audio2_path)
            
            # 가중 평균으로 블렌딩
            blended_x = x1 * (1 - ratio) + x2 * ratio
            
            lang = None if language == "auto" else language
            
            wavs, sr = tts_model.generate_voice_clone(
                text=text,
                language=lang,
                x_vector=blended_x,
            )
        except (AttributeError, TypeError):
            # 방법 2: x_vector_only_mode 사용
            print("[Blend] Trying x_vector_only_mode approach...")
            
            # 두 음성으로 각각 생성 후 오디오 레벨에서 블렌딩
            lang = None if language == "auto" else language
            
            wavs1, sr = tts_model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=audio1_path,
                x_vector_only_mode=True,
            )
            
            wavs2, sr = tts_model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=audio2_path,
                x_vector_only_mode=True,
            )
            
            # 오디오 레벨 블렌딩
            wav1 = np.array(wavs1[0])
            wav2 = np.array(wavs2[0])
            
            # 길이 맞추기
            min_len = min(len(wav1), len(wav2))
            wav1 = wav1[:min_len]
            wav2 = wav2[:min_len]
            
            # 블렌딩
            wavs = [wav1 * (1 - ratio) + wav2 * ratio]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"blend_{int(ratio*100)}_{timestamp}.wav"
        
        sf.write(str(output_path), wavs[0], sr)
        
        return str(output_path), f"✅ Blended: Voice1 {int((1-ratio)*100)}% + Voice2 {int(ratio*100)}%"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

def blend_voices_v2(audio1, audio2, audio3, weight1, weight2, weight3, text, language="auto", save_as_preset=""):
    """
    Resemblyzer를 사용한 진정한 Voice Blending
    - 2개 또는 3개 음성 지원 (Voice 3는 선택적)
    - Speaker embedding 레벨에서 SLERP 블렌딩
    - 블렌딩된 음성을 Qwen3-TTS 프리셋으로 저장 가능
    """
    global tts_model, current_model_type
    
    import soundfile as sf
    import tempfile
    import numpy as np
    
    # 입력 검증 - 최소 2개 음성 필요
    if audio1 is None or audio2 is None:
        return None, "❌ 최소 2개 음성이 필요합니다 (Voice 1, Voice 2)"
    
    if not text.strip():
        return None, "❌ 텍스트를 입력하세요"
    
    # 오디오 경로 처리 헬퍼
    def get_audio_path(audio_input):
        if audio_input is None:
            return None
        if isinstance(audio_input, tuple):
            sr, data = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, data, sr)
                return f.name
        return audio_input
    
    audio1_path = get_audio_path(audio1)
    audio2_path = get_audio_path(audio2)
    audio3_path = get_audio_path(audio3)  # None이면 2개만 블렌딩
    
    # 사용할 음성 수 결정
    use_3_voices = audio3_path is not None
    
    try:
        # ============ Resemblyzer로 Speaker Embedding 추출 ============
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            print("[Blend] Using Resemblyzer for speaker embedding extraction")
            
            encoder = VoiceEncoder()
            
            # 음성 전처리 및 embedding 추출
            wav1 = preprocess_wav(audio1_path)
            wav2 = preprocess_wav(audio2_path)
            
            embed1 = encoder.embed_utterance(wav1)
            embed2 = encoder.embed_utterance(wav2)
            
            if use_3_voices:
                wav3 = preprocess_wav(audio3_path)
                embed3 = encoder.embed_utterance(wav3)
            
            print(f"[Blend] Embedding shape: {embed1.shape}")
            
        except ImportError:
            return None, "❌ Resemblyzer 설치 필요: pip install resemblyzer"
        
        # ============ 가중치 정규화 ============
        if use_3_voices:
            total = weight1 + weight2 + weight3
            if total <= 0:
                return None, "❌ 가중치 합이 0 이상이어야 합니다"
            w1 = weight1 / total
            w2 = weight2 / total
            w3 = weight3 / total
            print(f"[Blend] 3-Way: V1={w1:.2f}, V2={w2:.2f}, V3={w3:.2f}")
        else:
            total = weight1 + weight2
            if total <= 0:
                return None, "❌ 가중치 합이 0 이상이어야 합니다"
            w1 = weight1 / total
            w2 = weight2 / total
            w3 = 0
            print(f"[Blend] 2-Way: V1={w1:.2f}, V2={w2:.2f}")
        
        # ============ SLERP 블렌딩 ============
        def slerp(v0, v1, t):
            """Spherical Linear Interpolation - 고품질 벡터 블렌딩"""
            v0 = v0 / np.linalg.norm(v0)
            v1 = v1 / np.linalg.norm(v1)
            
            dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
            theta = np.arccos(dot)
            
            if theta < 1e-6:
                return v0 * (1 - t) + v1 * t
            
            sin_theta = np.sin(theta)
            return (np.sin((1 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1
        
        if use_3_voices:
            # 3개 블렌딩: 먼저 1,2를 블렌딩하고, 그 결과와 3을 블렌딩
            # 또는 가중 평균 사용 (더 직관적)
            blended_embed = embed1 * w1 + embed2 * w2 + embed3 * w3
            blended_embed = blended_embed / np.linalg.norm(blended_embed)  # 정규화
        else:
            # 2개 블렌딩: SLERP 사용
            blended_embed = slerp(embed1, embed2, w2)
        
        print(f"[Blend] Blended embedding created (for analysis)")
        
        # ============ 참조 오디오 연결 (Concatenate) ============
        # 여러 오디오를 가중치에 따라 연결하여 하나의 참조로 사용
        # Qwen이 자체적으로 혼합된 speaker 특성을 추출하게 함
        
        import librosa
        
        # 각 음성 로드
        y1, sr1 = librosa.load(audio1_path, sr=None)
        y2, sr2 = librosa.load(audio2_path, sr=None)
        
        target_sr = 24000
        if sr1 != target_sr:
            y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sr)
        if sr2 != target_sr:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sr)
        
        if use_3_voices:
            y3, sr3 = librosa.load(audio3_path, sr=None)
            if sr3 != target_sr:
                y3 = librosa.resample(y3, orig_sr=sr3, target_sr=target_sr)
        
        # 가중치에 따라 각 오디오의 길이(비중) 결정
        # 예: 50%/50% → 각각 절반씩, 70%/30% → 첫 번째가 더 길게
        base_length = 3 * target_sr  # 3초 기준
        
        len1 = int(base_length * w1)
        len2 = int(base_length * w2)
        
        # 각 오디오에서 가중치 비율만큼 추출 (앞부분)
        segment1 = y1[:min(len1, len(y1))]
        segment2 = y2[:min(len2, len(y2))]
        
        if use_3_voices:
            len3 = int(base_length * w3)
            segment3 = y3[:min(len3, len(y3))]
            # 연결
            combined_audio = np.concatenate([segment1, segment2, segment3])
            print(f"[Blend] Combined 3 voices: {len(segment1)/target_sr:.1f}s + {len(segment2)/target_sr:.1f}s + {len(segment3)/target_sr:.1f}s")
        else:
            # 연결
            combined_audio = np.concatenate([segment1, segment2])
            print(f"[Blend] Combined 2 voices: {len(segment1)/target_sr:.1f}s + {len(segment2)/target_sr:.1f}s")
        
        # 연결된 참조 오디오 저장
        combined_ref_path = OUTPUT_DIR / f"blend_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(str(combined_ref_path), combined_audio.astype(np.float32), target_sr)
        print(f"[Blend] Created combined reference: {combined_ref_path}")
        
        # ============ Qwen3-TTS로 TTS 생성 (한 번만!) ============
        if current_model_type != "base":
            status = load_model("base")
            if "❌" in status:
                return None, status
        
        lang = None if language == "auto" else language
        
        print("[Blend] Generating TTS with combined reference...")
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language=lang,
            ref_audio=str(combined_ref_path),
            x_vector_only_mode=True,  # 음색만 추출
        )
        
        print("[Blend] TTS generation completed")
        
        # 출력 저장
        global last_blend_output_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if use_3_voices:
            output_path = OUTPUT_DIR / f"blend3_{int(w1*100)}_{int(w2*100)}_{int(w3*100)}_{timestamp}.wav"
            blend_info = f"V1={int(w1*100)}% + V2={int(w2*100)}% + V3={int(w3*100)}%"
        else:
            output_path = OUTPUT_DIR / f"blend2_{int(w1*100)}_{int(w2*100)}_{timestamp}.wav"
            blend_info = f"V1={int(w1*100)}% + V2={int(w2*100)}%"
        
        sf.write(str(output_path), wavs[0], sr)
        
        # 블렌딩 참조 오디오 경로 저장 (프리셋 저장용)
        last_blend_output_path = str(combined_ref_path)
        
        # ============ 프리셋으로 저장 (선택적) ============
        if save_as_preset and save_as_preset.strip():
            preset_name = save_as_preset.strip()
            presets = get_presets()
            
            # 블렌딩된 참조 음성을 프리셋으로 복사
            import shutil
            preset_audio_path = PRESETS_DIR / f"{preset_name}.wav"
            shutil.copy(str(combined_ref_path), str(preset_audio_path))
            
            presets[preset_name] = {
                "audio": str(preset_audio_path),
                "transcript": "",
                "created": datetime.now().isoformat(),
                "type": "blended",
                "blend_info": blend_info
            }
            save_presets(presets)
            
            return str(output_path), f"✅ Blend 완료: {blend_info}\n💾 프리셋 '{preset_name}' 저장됨"
        
        return str(output_path), f"✅ Blend 완료: {blend_info}"
    
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n{traceback.format_exc()}"

# 마지막 블렌드 결과 저장용 전역 변수
last_blend_output_path = None

def save_blend_as_preset(preset_name):
    """마지막 블렌드 결과를 프리셋으로 저장"""
    global last_blend_output_path
    
    print(f"[SavePreset] Called with name: '{preset_name}'")
    print(f"[SavePreset] last_blend_output_path: {last_blend_output_path}")
    
    if not preset_name or not preset_name.strip():
        print("[SavePreset] Error: No preset name")
        return "❌ 프리셋 이름을 입력하세요"
    
    if last_blend_output_path is None:
        print("[SavePreset] Error: No blend path saved")
        return "❌ 먼저 블렌딩을 실행하세요"
    
    if not Path(last_blend_output_path).exists():
        print(f"[SavePreset] Error: File not found: {last_blend_output_path}")
        return "❌ 블렌딩 파일을 찾을 수 없습니다"
    
    preset_name = preset_name.strip()
    
    try:
        import shutil
        
        # 프리셋으로 복사
        preset_audio_path = PRESETS_DIR / f"{preset_name}.wav"
        shutil.copy(last_blend_output_path, str(preset_audio_path))
        print(f"[SavePreset] Copied to: {preset_audio_path}")
        
        # 프리셋 정보 저장
        presets = get_presets()
        presets[preset_name] = {
            "audio": str(preset_audio_path),
            "transcript": "",
            "created": datetime.now().isoformat(),
            "type": "blended"
        }
        save_presets(presets)
        
        print(f"[SavePreset] Success: {preset_name}")
        return f"✅ 프리셋 '{preset_name}' 저장 완료! Voice Clone 탭에서 사용 가능"
    
    except Exception as e:
        print(f"[SavePreset] Exception: {e}")
        return f"❌ 저장 실패: {e}"

# ============ RVC ============
def get_rvc_models():
    models = [f.stem for f in RVC_MODELS_DIR.glob("**/*.pth")]
    return models if models else ["No models - add .pth files"]

def rvc_convert(input_audio, model_name, pitch=0):
    if input_audio is None:
        return None, "Provide input audio"
    
    if not model_name or "No models" in model_name:
        return None, "Add RVC models to rvc_models folder"
    
    try:
        from rvc_python import RVCInference
        import soundfile as sf
        import tempfile
        
        # Gradio audio 처리
        if isinstance(input_audio, tuple):
            sr, audio_data = input_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_data, sr)
            audio_path = temp_path
            print(f"[RVC] Converted tuple to: {temp_path}")
        else:
            audio_path = input_audio
        
        print(f"[RVC] Input: {audio_path}")
        print(f"[RVC] Model: {model_name}")
        
        rvc = RVCInference(device="cuda:0")
        
        model_path = None
        for f in RVC_MODELS_DIR.glob(f"**/{model_name}.pth"):
            model_path = f
            break
        
        if not model_path:
            return None, f"Model not found: {model_name}"
        
        print(f"[RVC] Loading model: {model_path}")
        rvc.load_model(str(model_path))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"rvc_{model_name}_{timestamp}.wav"
        
        print(f"[RVC] Converting...")
        rvc.infer_file(audio_path, str(output_path), pitch=pitch)
        
        print(f"[RVC] Done: {output_path}")
        return str(output_path), f"RVC conversion done"
    
    except Exception as e:
        import traceback
        print(f"[RVC] Error: {traceback.format_exc()}")
        return None, f"RVC Error: {e}"

# ============ Gradio UI ============

# Custom HTML/JS for playback speed control - 0.1 단위
CUSTOM_HEAD = """
<style>
.custom-speed-select {
    background: #374151 !important;
    color: #fff !important;
    border: 1px solid #4b5563 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 12px !important;
    cursor: pointer !important;
    min-width: 55px !important;
}
</style>
<script>
(function() {
    var currentSpeed = 1.0;
    
    // 배속 적용 함수
    function applySpeed(rate) {
        currentSpeed = rate;
        // 모든 audio 요소에 적용
        var audios = document.getElementsByTagName('audio');
        for (var i = 0; i < audios.length; i++) {
            audios[i].playbackRate = rate;
        }
        console.log('Speed applied:', rate, 'to', audios.length, 'audio elements');
    }
    
    // 새 오디오가 재생 시작할 때마다 배속 적용
    document.addEventListener('play', function(e) {
        if (e.target.tagName === 'AUDIO') {
            e.target.playbackRate = currentSpeed;
            console.log('Speed set on play:', currentSpeed);
        }
    }, true);
    
    // 배속 버튼 교체 함수
    function replaceSpeedButtons() {
        // Gradio의 배속 버튼 찾기 (1x, 1.5x, 2x 패턴)
        var allButtons = document.querySelectorAll('button');
        allButtons.forEach(function(btn) {
            var text = (btn.textContent || btn.innerText || '').trim();
            // 배속 버튼 패턴: 숫자x 또는 숫자.숫자x
            if (/^[0-2](\.[0-9])?x$/.test(text) && !btn.dataset.replaced) {
                btn.dataset.replaced = 'true';
                
                // 드롭다운 생성
                var select = document.createElement('select');
                select.className = 'custom-speed-select';
                
                var speeds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
                speeds.forEach(function(speed) {
                    var opt = document.createElement('option');
                    opt.value = speed;
                    opt.textContent = speed.toFixed(1) + 'x';
                    if (speed === 1.0) opt.selected = true;
                    select.appendChild(opt);
                });
                
                select.addEventListener('change', function(e) {
                    applySpeed(parseFloat(e.target.value));
                });
                
                // 버튼을 드롭다운으로 교체
                if (btn.parentNode) {
                    btn.parentNode.replaceChild(select, btn);
                }
            }
        });
    }
    
    // 주기적 실행
    setInterval(replaceSpeedButtons, 1500);
    setInterval(function() { applySpeed(currentSpeed); }, 500);
    
    // DOM 변경 감지
    var observer = new MutationObserver(function() {
        setTimeout(replaceSpeedButtons, 300);
    });
    
    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
</script>
"""

def create_ui():
    with gr.Blocks(title="Voice Manager", theme=gr.themes.Soft(), head=CUSTOM_HEAD) as app:
        gr.Markdown("# 🎙️ Voice Manager")
        gr.Markdown("Qwen3-TTS + RVC 통합 음성 생성")
        
        status = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        with gr.Tab("🔧 Setup"):
            gr.Markdown("### 모델 관리")
            gr.Markdown("모델은 각 탭 사용 시 자동 로드됩니다")
            
            with gr.Row():
                load_custom_btn = gr.Button("Load CustomVoice")
                load_design_btn = gr.Button("Load VoiceDesign")
                load_base_btn = gr.Button("Load Base (Clone)")
                unload_btn = gr.Button("Unload All", variant="stop")
            
            load_custom_btn.click(lambda: load_model("custom"), outputs=status)
            load_design_btn.click(lambda: load_model("design"), outputs=status)
            load_base_btn.click(lambda: load_model("base"), outputs=status)
            unload_btn.click(unload_model, outputs=status)
            
            gr.Markdown("### GPU 상태")
            vram_info = gr.Textbox(label="Info", interactive=False)
            refresh_btn = gr.Button("🔄 Refresh")
            refresh_btn.click(get_vram_info, outputs=vram_info)
        
        with gr.Tab("👤 CustomVoice"):
            gr.Markdown("### 9개 프리셋 보이스 사용")
            
            # 스피커 미리듣기 섹션
            gr.Markdown("#### 🎧 스피커 미리듣기")
            with gr.Row():
                preview_speaker = gr.Dropdown(
                    label="Speaker 선택", 
                    choices=CUSTOM_SPEAKERS, 
                    value="vivian"
                )
                preview_btn = gr.Button("▶️ 미리듣기", variant="secondary")
            
            with gr.Row():
                preview_audio = gr.Audio(label="미리듣기", interactive=False)
                preview_info = gr.Textbox(label="스피커 정보", interactive=False)
            
            preview_btn.click(
                generate_speaker_preview,
                inputs=[preview_speaker],
                outputs=[preview_audio, preview_info]
            )
            
            gr.Markdown("---")
            gr.Markdown("#### 🎤 음성 생성")
            
            cv_speaker = gr.Dropdown(label="Speaker", choices=CUSTOM_SPEAKERS, value="vivian")
            cv_text = gr.Textbox(label="Text", lines=3, placeholder="Hello, this is a test.")
            
            gr.Markdown("#### 🎭 감정/스타일 컨트롤")
            with gr.Row():
                cv_emotion = gr.Dropdown(
                    label="감정",
                    choices=["(선택안함)", "happy", "sad", "angry", "excited", "calm", "fearful", "surprised", "tender", "serious"],
                    value="(선택안함)"
                )
                cv_speed = gr.Dropdown(
                    label="속도",
                    choices=["(선택안함)", "very slow", "slow", "normal", "fast", "very fast"],
                    value="(선택안함)"
                )
                cv_tone = gr.Dropdown(
                    label="톤",
                    choices=["(선택안함)", "warm", "cold", "bright", "dark", "soft", "firm"],
                    value="(선택안함)"
                )
            
            cv_instruct = gr.Textbox(
                label="커스텀 Instruct (위 옵션 대신 직접 입력)",
                placeholder="e.g., whisper gently, speak with enthusiasm, dramatic pause between sentences",
                lines=2
            )
            gr.Markdown("💡 **팁**: 드롭다운 선택하면 자동 조합됨. 커스텀 입력하면 그게 우선됨.")
            
            cv_btn = gr.Button("🔊 Generate", variant="primary")
            cv_output = gr.Audio(label="Output")
            
            def generate_with_style(text, speaker, emotion, speed, tone, custom_instruct):
                # 커스텀 입력이 있으면 그걸 사용
                if custom_instruct.strip():
                    instruct = custom_instruct.strip()
                else:
                    # 드롭다운 조합
                    parts = []
                    if emotion != "(선택안함)":
                        parts.append(emotion)
                    if speed != "(선택안함)":
                        parts.append(f"speak {speed}")
                    if tone != "(선택안함)":
                        parts.append(f"{tone} tone")
                    instruct = ", ".join(parts) if parts else ""
                
                return generate_custom_voice(text, speaker, instruct)
            
            cv_btn.click(
                generate_with_style,
                inputs=[cv_text, cv_speaker, cv_emotion, cv_speed, cv_tone, cv_instruct],
                outputs=[cv_output, status]
            )
        
        with gr.Tab("✨ VoiceDesign"):
            gr.Markdown("### 텍스트 설명으로 새 목소리 생성")
            
            vd_text = gr.Textbox(label="Text to Speak", lines=3)
            vd_desc = gr.Textbox(
                label="Voice Description",
                lines=2,
                placeholder="A deep male voice with British accent, calm and professional"
            )
            
            vd_btn = gr.Button("🔊 Generate", variant="primary")
            vd_output = gr.Audio(label="Output")
            
            vd_btn.click(
                generate_voice_design,
                inputs=[vd_text, vd_desc],
                outputs=[vd_output, status]
            )
        
        with gr.Tab("🎬 멀티라인 TTS"):
            gr.Markdown("### 대사별 감정/스타일 지정 (CustomVoice 전용)")
            gr.Markdown("""
**형식:** `[감정] 대사` 또는 `[감정, 속도] 대사`

**예시:**
```
[신남] 안녕하세요! 만나서 반가워요!
[슬픔, 느리게] 정말 슬픈 일이에요...
[화남] 이게 대체 뭐야!
[happy, fast] Wow, this is amazing!
[calm, warm] 괜찮아요, 천천히 해도 돼요.
[whisper] 쉿, 조용히...
```

**사용 가능한 태그 (영어/한글):**
| 영어 | 한글 |
|------|------|
| happy | 신남, 기쁨, 행복 |
| sad | 슬픔, 우울 |
| angry | 화남, 분노 |
| excited | 흥분, 신남 |
| calm | 차분, 평온 |
| whisper | 속삭임 |
| slow | 느리게 |
| fast | 빠르게 |

⚠️ **Voice Clone은 감정 컨트롤 미지원** - CustomVoice만 사용하세요
""")
            
            ml2_speaker = gr.Dropdown(label="Speaker", choices=CUSTOM_SPEAKERS, value="sohee")
            ml2_text = gr.Textbox(
                label="Script (줄마다 태그)", 
                lines=10,
                placeholder="[신남] 안녕하세요!\n[슬픔, 느리게] 정말 슬픈 일이에요...\n[화남] 이게 대체 뭐야!\n[calm] 괜찮아요."
            )
            ml2_btn = gr.Button("🔊 Generate All", variant="primary")
            ml2_output = gr.Audio(label="Output")
            
            ml2_btn.click(generate_multiline_custom, [ml2_text, ml2_speaker], [ml2_output, status])
        
        with gr.Tab("🎭 Expressive Clone"):
            gr.Markdown("### 내 목소리 + 감정 표현")
            gr.Markdown("""
**내 프리셋(블렌딩 포함) 목소리**로 **감정 표현**까지!

**사용법:**
1. 프리셋 선택 (Voice Blend에서 만든 것도 OK)
2. `[감정] 대사` 형식으로 입력
3. Generate 클릭

**예시:**
```
[신남] 안녕하세요! 만나서 반가워요!
[슬픔] 정말 아쉽네요...
[화남] 이건 정말 아니야!
[속삭임] 쉿, 조용히...
```
""")
            
            ec_preset = gr.Dropdown(label="내 프리셋 선택", choices=get_preset_choices())
            ec_text = gr.Textbox(
                label="Script (감정 태그 + 대사)", 
                lines=8,
                placeholder="[신남] 안녕하세요!\n[슬픔, 느리게] 정말 슬픈 일이에요...\n[화남] 이게 뭐야!"
            )
            ec_lang = gr.Dropdown(
                label="Language",
                choices=["auto", "Korean", "English", "Chinese", "Japanese"],
                value="auto"
            )
            
            ec_refresh = gr.Button("🔄 Refresh Presets")
            ec_btn = gr.Button("🎭 Generate Expressive", variant="primary")
            ec_output = gr.Audio(label="Output")
            
            ec_refresh.click(lambda: gr.update(choices=get_preset_choices()), outputs=ec_preset)
            ec_btn.click(generate_expressive_clone, [ec_text, ec_preset, ec_lang], [ec_output, status])
        
        with gr.Tab("🎙️ Voice Dub"):
            gr.Markdown("### 참조 음성 → 내 목소리로 더빙")
            gr.Markdown("""
**2단계 워크플로우:**
1. 참조 음성 업로드 → **텍스트 추출** 버튼
2. 추출된 텍스트 수정 (무음 추가 가능: `[pause:0.5]`)
3. **Generate Dub** 버튼

**무음 태그:** `[pause:초]` - 예: `[pause:0.5]` = 0.5초 무음
""")
            
            dub_ref_audio = gr.Audio(label="🎧 참조 음성 (더빙할 원본)", type="filepath")
            
            with gr.Row():
                dub_extract_btn = gr.Button("📝 텍스트 추출 (Whisper)", variant="secondary")
            
            dub_transcript = gr.Textbox(
                label="📝 텍스트 (수정 가능)", 
                lines=4,
                placeholder="텍스트 추출 후 여기서 수정하세요...\n[pause:0.5] 태그로 무음 추가 가능",
                interactive=True
            )
            
            dub_prompt = gr.Dropdown(label="🎤 저장된 프롬프트", choices=get_prompt_choices())
            dub_lang = gr.Dropdown(
                label="Language",
                choices=["auto", "Korean", "English", "Chinese", "Japanese"],
                value="auto"
            )
            
            with gr.Row():
                dub_refresh = gr.Button("🔄 Refresh Prompts")
                dub_btn = gr.Button("🎬 Generate Dub", variant="primary")
            
            dub_output = gr.Audio(label="🔊 더빙 결과")
            
            dub_extract_btn.click(transcribe_audio, [dub_ref_audio], [dub_transcript])
            dub_refresh.click(lambda: gr.update(choices=get_prompt_choices()), outputs=dub_prompt)
            dub_btn.click(
                generate_voice_dub_with_text, 
                [dub_transcript, dub_prompt, dub_lang], 
                [dub_output, status]
            )
        
        with gr.Tab("📦 Voice Prompt"):
            gr.Markdown("### Voice Clone Prompt 관리")
            gr.Markdown("""
**Pre-compute Prompt** (ComfyUI 방식)
- 참조 음성에서 음색+운율 정보를 미리 추출하여 저장
- 저장된 프롬프트로 빠르게 TTS 생성 가능
""")
            
            # 마이그레이션 버튼
            with gr.Accordion("🔄 기존 Presets 마이그레이션", open=False):
                gr.Markdown("기존 `presets/` 폴더의 프리셋을 프롬프트로 변환합니다.")
                migrate_btn = gr.Button("📦 Presets → Prompts 마이그레이션", variant="secondary")
                migrate_btn.click(migrate_presets_to_prompts, outputs=status)
            
            gr.Markdown("---")
            
            with gr.Row():
                vp_name = gr.Textbox(label="프롬프트 이름", placeholder="my_voice_prompt")
                vp_audio = gr.Audio(label="참조 음성 (3-15초)", type="filepath")
            
            with gr.Row():
                vp_text = gr.Textbox(label="참조 텍스트 (권장)", placeholder="참조 음성의 내용", scale=4)
                vp_stt_btn = gr.Button("🎤 Whisper 추출", scale=1)
            
            vp_stt_btn.click(transcribe_audio, [vp_audio], [vp_text])
            
            with gr.Row():
                vp_create_btn = gr.Button("✨ Create Prompt", variant="primary")
                vp_dropdown = gr.Dropdown(label="저장된 프롬프트", choices=get_prompt_choices())
                vp_delete_btn = gr.Button("🗑️ Delete", variant="stop")
            
            vp_create_btn.click(create_and_save_prompt, [vp_name, vp_audio, vp_text], [status])
            vp_create_btn.click(lambda: gr.update(choices=get_prompt_choices()), outputs=vp_dropdown)
            vp_delete_btn.click(delete_prompt, [vp_dropdown], [status, vp_dropdown])
            
            gr.Markdown("---")
            gr.Markdown("### 프롬프트로 TTS 생성")
            
            vp_gen_prompt = gr.Dropdown(label="프롬프트 선택", choices=get_prompt_choices())
            vp_gen_text = gr.Textbox(label="텍스트", lines=3, placeholder="생성할 텍스트 입력...")
            vp_gen_lang = gr.Dropdown(
                label="Language",
                choices=["auto", "Korean", "English", "Chinese", "Japanese"],
                value="auto"
            )
            
            vp_gen_refresh = gr.Button("🔄 Refresh")
            vp_gen_btn = gr.Button("🔊 Generate", variant="primary")
            vp_gen_output = gr.Audio(label="Output")
            
            vp_gen_refresh.click(lambda: (gr.update(choices=get_prompt_choices()), gr.update(choices=get_prompt_choices())), outputs=[vp_dropdown, vp_gen_prompt])
            vp_gen_btn.click(generate_from_prompt, [vp_gen_text, vp_gen_prompt, vp_gen_lang], [vp_gen_output, status])
        
        with gr.Tab("🎤 Voice Clone"):
            gr.Markdown("### 프롬프트로 Voice Clone")
            gr.Markdown("💡 먼저 **Voice Prompt** 탭에서 목소리를 등록하세요")
            
            vc_prompt = gr.Dropdown(label="Select Prompt", choices=get_prompt_choices())
            vc_text = gr.Textbox(label="Text to Speak", lines=3)
            vc_language = gr.Dropdown(
                label="Language",
                choices=["auto", "Korean", "English", "Chinese", "Japanese", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                value="auto"
            )
            
            vc_refresh = gr.Button("🔄 Refresh Prompts")
            vc_btn = gr.Button("🔊 Generate", variant="primary")
            vc_output = gr.Audio(label="Output")
            
            vc_refresh.click(lambda: gr.update(choices=get_prompt_choices()), outputs=vc_prompt)
            vc_btn.click(
                generate_from_prompt, 
                [vc_text, vc_prompt, vc_language], 
                [vc_output, status]
            )
        
        with gr.Tab("🔀 Voice Blend"):
            gr.Markdown("### 🎭 음성 블렌딩 (Resemblyzer)")
            gr.Markdown("2~3개 화자의 음성 특성을 섞어서 **새로운 목소리** 생성")
            gr.Markdown("> 💡 Voice 3는 선택사항입니다. 2개만 업로드해도 블렌딩됩니다.")
            
            with gr.Row():
                blend_audio1 = gr.Audio(label="🔊 Voice 1 (필수)", type="filepath")
                blend_audio2 = gr.Audio(label="🔊 Voice 2 (필수)", type="filepath")
                blend_audio3 = gr.Audio(label="🔊 Voice 3 (선택)", type="filepath")
            
            gr.Markdown("#### ⚖️ 가중치 (합계 자동 정규화)")
            with gr.Row():
                blend_weight1 = gr.Slider(0, 100, value=50, step=1, label="Voice 1 가중치 (%)")
                blend_weight2 = gr.Slider(0, 100, value=50, step=1, label="Voice 2 가중치 (%)")
                blend_weight3 = gr.Slider(0, 100, value=0, step=1, label="Voice 3 가중치 (%)")
            
            blend_text = gr.Textbox(label="Text to Speak", lines=3, placeholder="블렌딩된 목소리로 말할 텍스트")
            blend_language = gr.Dropdown(
                label="Language",
                choices=["auto", "Korean", "English", "Chinese", "Japanese", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                value="auto"
            )
            
            blend_btn = gr.Button("🎭 Blend & Generate", variant="primary", size="lg")
            blend_output = gr.Audio(label="🎵 Blended Output")
            
            gr.Markdown("---")
            gr.Markdown("#### 💾 결과를 프리셋으로 저장 (듣고 나서 저장)")
            with gr.Row():
                blend_preset_name = gr.Textbox(
                    label="Preset Name", 
                    placeholder="저장할 프리셋 이름 입력",
                    scale=3
                )
                blend_save_btn = gr.Button("💾 프리셋 저장", variant="secondary", scale=1)
            
            blend_save_status = gr.Textbox(label="저장 상태", interactive=False)
            
            # 블렌딩 (프리셋 저장 없이)
            blend_btn.click(
                lambda a1, a2, a3, w1, w2, w3, t, l: blend_voices_v2(a1, a2, a3, w1, w2, w3, t, l, ""), 
                [blend_audio1, blend_audio2, blend_audio3, blend_weight1, blend_weight2, blend_weight3, blend_text, blend_language], 
                [blend_output, status]
            )
            
            # 결과를 프리셋으로 저장
            blend_save_btn.click(
                save_blend_as_preset,
                [blend_preset_name],
                [blend_save_status]
            )
        
        with gr.Tab("🎛️ RVC"):
            gr.Markdown("### RVC 음성 변환")
            gr.Markdown("⚠️ **RVC는 별도 환경 필요** (fairseq C++ 빌드 필요)")
            gr.Markdown("RVC 사용하려면 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI 참고")
            gr.Markdown(f"📂 모델 경로: `{RVC_MODELS_DIR}`")
            
            rvc_input = gr.Audio(label="Input Audio", type="filepath")
            rvc_model = gr.Dropdown(label="RVC Model", choices=get_rvc_models())
            rvc_pitch = gr.Slider(-12, 12, value=0, step=1, label="Pitch")
            
            rvc_refresh = gr.Button("🔄 Refresh Models")
            rvc_btn = gr.Button("🔊 Convert (현재 비활성)", variant="secondary", interactive=False)
            rvc_output = gr.Audio(label="Output")
            
            rvc_refresh.click(lambda: gr.update(choices=get_rvc_models()), outputs=rvc_model)
        
        with gr.Tab("📂 Outputs"):
            gr.Markdown(f"### 출력 폴더: `{OUTPUT_DIR}`")
            
            def list_outputs():
                files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
                return "\n".join([f"• {f.name}" for f in files[:20]]) or "No files yet"
            
            out_list = gr.Textbox(label="Recent Files", lines=15, interactive=False)
            out_btn = gr.Button("🔄 Refresh")
            out_btn.click(list_outputs, outputs=out_list)
    
    return app

if __name__ == "__main__":
    print("=" * 50)
    print("Voice Manager - Qwen3-TTS + RVC")
    print("=" * 50)
    print(f"Presets: {PRESETS_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"RVC: {RVC_MODELS_DIR}")
    print("=" * 50)
    
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True,
    )

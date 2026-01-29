# 🎙️ Voice Manager 사용 설명서

## 개요

Qwen3-TTS + RVC 통합 음성 생성 시스템

**실행**: `E:\ai_tool\tts_make\run_voice_manager.bat` 더블클릭
**URL**: <http://127.0.0.1:7861>

---

## 📁 폴더 구조

```
E:\ai_tool\tts_make\
├── .venv\              # Python 가상환경
├── presets\            # Voice Clone 프리셋 저장
├── outputs\            # 생성된 오디오 출력
├── rvc_models\         # RVC 모델 (.pth 파일)
├── voice_manager.py    # 메인 앱
└── run_voice_manager.bat
```

---

## 🔧 탭별 기능

### 1. Setup (설정)

- **모델 수동 로드/언로드**
- **VRAM 상태 확인**
- 💡 모델은 각 탭 사용 시 자동 로드됨

### 2. CustomVoice (프리셋 보이스)

9개 내장 음성으로 TTS 생성

| Speaker | 설명 |
|---------|------|
| `vivian` | 여성, 영어 |
| `serena` | 여성, 영어 |
| `sohee` | 여성, 한국어 |
| `ono_anna` | 여성, 일본어 |
| `ryan` | 남성, 영어 |
| `aiden` | 남성, 영어 |
| `dylan` | 남성, 영어 |
| `eric` | 남성, 영어 |
| `uncle_fu` | 남성, 중국어 |

**사용법**:

1. Speaker 선택
2. Text 입력
3. Style Instruct (선택) - 예: "Speak slowly", "Happy tone"
4. Generate 클릭

### 3. VoiceDesign (목소리 디자인)

텍스트 설명으로 새로운 목소리 생성

**Voice Description 예시**:

- "A deep male voice with British accent, calm and professional"
- "Young female voice, cheerful and energetic, slight Japanese accent"
- "Old man voice, warm and gentle, storytelling style"
- "차분하고 부드러운 30대 여성 목소리"

**사용법**:

1. Text 입력 (말할 내용)
2. Voice Description 입력 (원하는 목소리 설명)
3. Generate 클릭

### 4. Presets (프리셋 관리)

Voice Clone용 음성 샘플 등록

**좋은 프리셋 조건**:

- 길이: 3~15초 (5~10초 권장)
- 품질: 깨끗한 녹음, 배경 소음 없음
- 내용: 자연스러운 말하기

**등록 방법**:

1. Name 입력 (예: my_voice)
2. Reference Audio 업로드
3. Transcript 입력 (음성 내용 텍스트) - **권장!**
4. Add 클릭

### 5. Voice Clone (음성 복제)

등록된 프리셋으로 TTS 생성

**사용법**:

1. 🔄 Refresh Presets 클릭
2. 프리셋 선택
3. Text 입력
4. Generate 클릭

### 6. Voice Blend (음성 블렌딩) 🆕

3개 화자의 음성 특성을 섞어서 **새로운 목소리** 생성

**특징**:

- Speaker Embedding (x_vector) 레벨 블렌딩
- 단순 오디오 믹싱이 아닌 진정한 새 목소리 생성
- 각 음성별 가중치 조절 가능 (자동 정규화)

**사용법**:

1. Voice 1, 2, 3에 각각 음성 샘플 업로드
2. 각 음성의 가중치 조절 (예: 50%, 30%, 20%)
3. Text 입력
4. Blend & Generate 클릭

### 7. RVC (음성 변환)

RVC 모델로 음성 톤/음색 변환

**모델 추가**:

1. `.pth` 파일을 `E:\ai_tool\tts_make\rvc_models\` 에 복사
2. 앱에서 🔄 Refresh Models 클릭

**사용법**:

1. Input Audio 업로드 (Qwen TTS 출력 또는 다른 음성)
2. RVC Model 선택
3. Pitch 조절 (-12 ~ +12 반음)
4. Convert 클릭

### 7. Outputs (출력 파일)

생성된 파일 목록 확인

---

## 💾 VRAM 사용량 (RTX 3090 24GB 기준)

| 모델 | VRAM |
|------|------|
| CustomVoice 1.7B | ~4-5GB |
| VoiceDesign 1.7B | ~4-5GB |
| Base (Clone) 1.7B | ~4-5GB |
| RVC | ~2GB |

⚠️ 한 번에 하나의 Qwen 모델만 로드됨 (자동 교체)

---

## 🎯 추천 워크플로우

### 빠른 TTS (내장 보이스)

1. CustomVoice 탭
2. 스피커 선택 → 텍스트 입력 → Generate

### 커스텀 목소리 (설명 기반)

1. VoiceDesign 탭
2. 원하는 목소리 설명 → Generate

### 특정 인물 목소리 복제

1. Presets 탭 → 샘플 등록
2. Voice Clone 탭 → 프리셋으로 생성

### 최종 품질 향상 (RVC)

1. Qwen TTS로 음성 생성
2. RVC 탭 → 음색 변환

---

## ⚠️ 문제 해결

### "SoX could not be found"

- 무시해도 됨 (경고일 뿐)

### 모델 로딩 느림

- 첫 실행 시 HuggingFace에서 다운로드 (~3-4GB)
- 이후에는 캐시 사용

### CUDA Out of Memory

- Setup 탭 → Unload All 클릭
- 다른 GPU 사용 앱 종료

### 포트 충돌 (7861)

- `voice_manager.py` 열고 `server_port=7862` 로 변경

---

## 📝 지원 언어

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

---

## 🔗 참고

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

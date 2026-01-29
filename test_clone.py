# Test Voice Clone - E:\ai_tool\tts_make\test_clone.py
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

ref_audio = r"E:\ai_tool\tts_make\presets\mig.wav"
ref_text = "솔직히 말씀드리면은 저한테 특별한 재능이 있어서가 아니거든요"

print("Generating voice clone...")
wavs, sr = model.generate_voice_clone(
    text="완료! 앱 재시작해주세요",
    language="Korean",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

output_path = r"E:\ai_tool\tts_make\outputs\test_clone_output.wav"
sf.write(output_path, wavs[0], sr)
print(f"Saved to: {output_path}")
print("Done!")

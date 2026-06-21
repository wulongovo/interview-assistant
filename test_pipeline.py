import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.insert(0, r"E:\ai-works\interview-assistant")

import numpy as np
import sounddevice as sd

print("=" * 50)
print("步骤1: 测试麦克风录音（录5秒，请说话）")
print("=" * 50)
sr = 16000
duration = 5
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
vol = np.abs(audio).mean()
print(f"  平均音量: {vol:.6f}")
print(f"  最大音量: {np.abs(audio).max():.6f}")

if vol < 0.005:
    print("  音量太低，没有检测到声音")
    sys.exit(1)
else:
    print("  检测到声音!")

audio_flat = audio.flatten()

print()
print("=" * 50)
print("步骤2: 测试 faster-whisper 语音识别")
print("=" * 50)
try:
    from faster_whisper import WhisperModel
    print("  正在加载模型...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("  模型加载成功!")

    print("  正在识别...")
    segs, info = model.transcribe(audio_flat, language="zh")
    texts = [s.text for s in segs]
    text = " ".join(texts).strip()
    print(f"  识别结果: '{text}'")
    if text:
        print("  语音识别成功!")
    else:
        print("  没有识别出文字")
except Exception as e:
    print(f"  识别失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("步骤3: 测试 Ollama 对话")
print("=" * 50)
try:
    import requests
    r = requests.post(
        "http://localhost:11434/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={"model": "qwen35-heretic", "messages": [{"role": "user", "content": text or "你好"}], "max_tokens": 100},
        timeout=30
    )
    if r.status_code == 200:
        ans = r.json()["choices"][0]["message"]["content"]
        print(f"  回答: {ans[:200]}")
        print("  对话成功!")
    else:
        print(f"  对话失败: {r.status_code}")
except Exception as e:
    print(f"  对话失败: {e}")

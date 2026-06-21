import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import time
import numpy as np

print("加载 faster-whisper...", flush=True)
t0 = time.time()
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
print(f"模型加载完成: {time.time()-t0:.1f}秒", flush=True)

audio = np.zeros(16000, dtype=np.float32)
print("识别中...", flush=True)
t0 = time.time()
segs, info = model.transcribe(audio, language="zh")
texts = [s.text for s in segs]
print(f"识别耗时: {time.time()-t0:.1f}秒", flush=True)
print(f"结果: '{' '.join(texts).strip()}'", flush=True)
print("完成!", flush=True)

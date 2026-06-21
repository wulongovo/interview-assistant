import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
import time

print("正在下载 faster-whisper-base 模型...", flush=True)
print("镜像: hf-mirror.com", flush=True)
t0 = time.time()
path = snapshot_download("Systran/faster-whisper-base")
print(f"下载完成! 耗时: {time.time()-t0:.1f}秒", flush=True)
print(f"路径: {path}", flush=True)

# Verify
from faster_whisper import WhisperModel
print("加载模型验证...", flush=True)
model = WhisperModel(path, device="cpu", compute_type="int8")
print("模型加载成功!", flush=True)

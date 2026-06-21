"""核心引擎：音频采集 + 语音识别 + AI回答"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import io
import wave
import time
import threading
import numpy as np
import sounddevice as sd
import requests
from config import load_config


class InterviewEngine:
    def __init__(self, on_transcript=None, on_answer=None, on_status=None):
        self.cfg = load_config()
        self.on_transcript = on_transcript
        self.on_answer = on_answer
        self.on_status = on_status
        self.running = False
        self.paused = False
        self.mode = "online"
        self.audio_device = None
        self._stream = None
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()
        self._process_thread = None
        self._history = []
        self.audio_level = 0.0
        self._recording = False
        self._record_stream = None
        self._record_buffer = []

    def list_audio_devices(self):
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append({"index": i, "name": d["name"]})
        return result

    def start(self):
        if self.running:
            return
        self.running = True
        self.paused = False
        self._audio_buffer = []
        self._update_status("监听中...")
        sr = self.cfg.get("sample_rate", 16000)
        self._stream = sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            device=self.audio_device,
            blocksize=int(sr * 0.5),
            callback=self._audio_callback
        )
        self._stream.start()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop(self):
        self.running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._update_status("已停止")

    def pause(self):
        self.paused = not self.paused
        self._update_status("已暂停" if self.paused else "监听中...")

    def ask_question(self, question):
        if not question or not question.strip():
            return
        threading.Thread(target=self._ask_thread, args=(question.strip(),), daemon=True).start()

    def _ask_thread(self, question):
        self._update_status("思考中...")
        if self.on_transcript:
            self.on_transcript(question)
        answer = self._get_answer(question)
        if self.on_answer:
            self.on_answer(answer)
        self._update_status("监听中..." if self.running else "就绪")

    def start_record(self):
        if self._recording:
            return
        self._recording = True
        self._record_buffer = []
        sr = self.cfg.get("sample_rate", 16000)
        self._record_stream = sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            device=self.audio_device,
            blocksize=int(sr * 0.1),
            callback=self._record_callback
        )
        self._record_stream.start()
        self._update_status("录音中...")

    def stop_record(self):
        if not self._recording:
            return ""
        self._recording = False
        if self._record_stream:
            self._record_stream.stop()
            self._record_stream.close()
            self._record_stream = None
        self.audio_level = 0.0

        with self._buffer_lock:
            if not self._record_buffer:
                self._update_status("未检测到声音")
                return ""
            audio_data = np.concatenate(self._record_buffer)
            self._record_buffer = []

        self._update_status("识别中...")
        try:
            text = self._transcribe_local(audio_data, self.cfg.get("sample_rate", 16000))
            self._update_status("识别完成")
            return text
        except Exception as e:
            self._update_status(f"识别失败: {e}")
            return ""

    def _record_callback(self, indata, frames, time_info, status):
        audio = indata.copy().flatten()
        self.audio_level = float(np.abs(audio).mean())
        with self._buffer_lock:
            self._record_buffer.append(audio)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running or self.paused:
            self.audio_level = 0.0
            return
        audio = indata.copy().flatten()
        self.audio_level = float(np.abs(audio).mean())
        if self.audio_level > self.cfg.get("silence_threshold", 0.015):
            with self._buffer_lock:
                self._audio_buffer.append(audio)

    def _process_loop(self):
        sr = self.cfg.get("sample_rate", 16000)
        sil_chunks = max(1, int(self.cfg.get("silence_duration", 1.2) / 0.5))
        silent_count = 0
        has_audio = False
        while self.running:
            time.sleep(0.2)
            if self.paused:
                continue
            with self._buffer_lock:
                if self._audio_buffer:
                    has_audio = True
                    silent_count = 0
                elif has_audio:
                    silent_count += 1
            if has_audio and silent_count >= sil_chunks:
                with self._buffer_lock:
                    audio_data = np.concatenate(self._audio_buffer)
                    self._audio_buffer = []
                has_audio = False
                silent_count = 0
                if len(audio_data) > sr * 0.5:
                    self._update_status("识别中...")
                    try:
                        text = self._transcribe(audio_data, sr)
                        if text and len(text.strip()) > 1:
                            if self.on_transcript:
                                self.on_transcript(text)
                            self._update_status("思考中...")
                            answer = self._get_answer(text)
                            if self.on_answer:
                                self.on_answer(answer)
                    except Exception as e:
                        if self.on_status:
                            self.on_status(f"识别错误: {e}")
                    self._update_status("监听中...")

    def _transcribe(self, audio_data, sr):
        if self.mode == "online":
            return self._transcribe_cloud(audio_data, sr)
        return self._transcribe_local(audio_data, sr)

    def _transcribe_cloud(self, audio_data, sr):
        try:
            buf = io.BytesIO()
            audio_i16 = (audio_data * 32767).astype(np.int16)
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_i16.tobytes())
            buf.seek(0)
            base = self.cfg.get("api_base", "https://api.openai.com/v1").rstrip("/")
            r = requests.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.cfg['api_key']}"},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={"model": "whisper-1", "language": self.cfg.get("language", "zh")},
                timeout=15
            )
            return r.json().get("text", "") if r.status_code == 200 else ""
        except Exception as e:
            return f"[云端识别错误: {e}]"

    def preload_local_model(self):
        if hasattr(self, "_wm"):
            return
        def _load():
            try:
                self._update_status("正在加载本地模型...")
                from faster_whisper import WhisperModel
                self._wm = WhisperModel("base", device="cpu", compute_type="int8")
                self._update_status("本地模型加载完成")
            except Exception as e:
                self._update_status(f"模型加载失败: {e}")
        threading.Thread(target=_load, daemon=True).start()

    def _transcribe_local(self, audio_data, sr):
        try:
            from faster_whisper import WhisperModel
            if not hasattr(self, "_wm"):
                self._update_status("加载本地模型（首次约10秒）...")
                self._wm = WhisperModel("base", device="cpu", compute_type="int8")
            self._update_status("识别中...")
            segs, _ = self._wm.transcribe(audio_data, language=self.cfg.get("language"))
            return " ".join(s.text for s in segs).strip()
        except ImportError:
            return "[需安装 faster-whisper]"
        except Exception as e:
            return f"[本地识别错误: {e}]"

    def _get_answer(self, question):
        try:
            self._history.append({"role": "user", "content": question})
            msgs = [{"role": "system", "content": self.cfg["system_prompt"]}]
            msgs.extend(self._history[-20:])
            base = self.cfg.get("api_base", "https://api.openai.com/v1").rstrip("/")
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg['api_key']}", "Content-Type": "application/json"},
                json={"model": self.cfg.get("model", "gpt-4o-mini"), "messages": msgs, "max_tokens": 500, "temperature": 0.7},
                timeout=60
            )
            if r.status_code == 200:
                ans = r.json()["choices"][0]["message"]["content"]
                self._history.append({"role": "assistant", "content": ans})
                return ans
            return f"[API错误 {r.status_code}]"
        except Exception as e:
            return f"[请求失败] {e}"

    def clear_history(self):
        self._history = []

    def reload_config(self):
        self.cfg = load_config()

    def _update_status(self, s):
        if self.on_status:
            self.on_status(s)

"""面试助手 - 主程序入口"""
import sys
import os
import json
import threading
import requests
import webview
from config import load_config, save_config
from engine import InterviewEngine

# 全局引用
window = None
engine = None


class Api:
    """暴露给前端 JS 调用的 Python API"""

    def __init__(self):
        self.engine = InterviewEngine(
            on_transcript=self._on_transcript,
            on_answer=self._on_answer,
            on_status=self._on_status
        )

    def _on_transcript(self, text):
        js = json.dumps({"type": "transcript", "data": text}, ensure_ascii=False)
        if window:
            window.evaluate_js(f"handleMessage({js})")

    def _on_answer(self, text):
        js = json.dumps({"type": "answer", "data": text}, ensure_ascii=False)
        if window:
            window.evaluate_js(f"handleMessage({js})")

    def _on_status(self, text):
        js = json.dumps({"type": "status", "data": text}, ensure_ascii=False)
        if window:
            window.evaluate_js(f"handleMessage({js})")

    def start_listen(self):
        self.engine.start()
        return {"ok": True}

    def stop_listen(self):
        self.engine.stop()
        return {"ok": True}

    def pause_listen(self):
        self.engine.pause()
        return {"ok": True, "paused": self.engine.paused}

    def clear_history(self):
        self.engine.clear_history()
        return {"ok": True}

    def send_text(self, text):
        """手动输入文字，直接调用AI回答"""
        if text and text.strip():
            self.engine.ask_question(text.strip())
            return {"ok": True}
        return {"ok": False, "msg": "请输入问题"}

    def get_audio_level(self):
        """获取当前音频音量"""
        return self.engine.audio_level

    def start_record(self):
        """开始录音"""
        self.engine.start_record()
        return {"ok": True}

    def stop_record(self):
        """停止录音并返回识别结果"""
        text = self.engine.stop_record()
        return {"ok": True, "text": text}

    def get_devices(self):
        return self.engine.list_audio_devices()

    def set_device(self, index):
        self.engine.audio_device = index
        return {"ok": True}

    def set_mode(self, mode):
        self.engine.mode = mode
        if mode == "offline":
            self.engine.preload_local_model()
        return {"ok": True}

    def get_config(self):
        return load_config()

    def save_settings(self, settings):
        cfg = load_config()
        cfg.update(settings)
        save_config(cfg)
        self.engine.reload_config()
        return {"ok": True}

    def test_api(self):
        """测试 API 连接"""
        cfg = load_config()
        if not cfg["api_key"]:
            return {"ok": False, "msg": "请先填写 API Key"}
        try:
            base = cfg["api_base"].rstrip("/")
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {cfg['api_key']}"}, timeout=10)
            if r.status_code == 200:
                return {"ok": True, "msg": "API 连接成功!"}
            return {"ok": False, "msg": f"API 返回 {r.status_code}"}
        except Exception as e:
            return {"ok": False, "msg": f"连接失败: {e}"}


def get_html_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")


def main():
    global window
    api = Api()

    window = webview.create_window(
        title="面试助手 Interview Assistant",
        url=get_html_path(),
        js_api=api,
        width=520,
        height=750,
        resizable=True,
        min_size=(400, 600),
        on_top=load_config().get("always_on_top", True),
        text_select=True
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()

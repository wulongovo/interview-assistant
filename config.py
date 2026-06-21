"""配置文件"""
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_CONFIG = {
    "api_key": "",
    "api_base": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "language": "zh",
    "sample_rate": 16000,
    "silence_threshold": 0.015,
    "silence_duration": 1.2,
    "system_prompt": """你是一个专业的面试助手。请根据面试官的问题，给出简洁、准确、专业的回答。\n\n要求：\n1. 回答要精炼，控制在3-5句话以内，方便用户快速阅读\n2. 如果是技术问题，给出核心要点和关键概念\n3. 如果是行为面试题，用STAR法则回答（情境-任务-行动-结果）\n4. 如果是薪资谈判，给出合理策略\n5. 语言与面试官提问语言一致（中文问中文答，英文问英文答）\n6. 不要用markdown格式，纯文本输出""",
    "always_on_top": True,
    "opacity": 0.95,
    "font_size": 14,
}

def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
            cfg.update(saved)
    return cfg

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

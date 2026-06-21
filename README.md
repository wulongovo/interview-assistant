# 🎤 面试助手 Interview Assistant

AI实时面试助手 - 语音识别 + 大模型，本地/云端双模式

## 功能

- 🎙 **语音识别** - faster-whisper 本地识别，离线可用
- 💬 **智能回答** - Ollama本地模型 / DeepSeek云端API
- 📊 **语音波动条** - 32条实时可视化，说话时绿色跳动
- ✏️ **手动输入** - 打字提问，按回车直接出回答
- 🔒 **隐私安全** - 全程本地运行，数据不上传

## 使用方法

```
conda activate pytorch
pip install -r requirements.txt
python main.py
```

或双击 `run.bat`

## 快捷键

- `Ctrl+Shift+S` 开始/停止监听
- `Ctrl+Shift+P` 暂停/继续
- `Ctrl+Shift+C` 清空记录
- `Enter` 发送输入

## 配置

在设置面板中配置：
- API Base URL (Ollama: http://localhost:11434/v1)
- 模型名称 (如 qwen35-heretic)
- 静音阈值
- 系统提示词

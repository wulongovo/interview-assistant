@echo off
chcp 65001 >nul
echo ==============================
echo   面试助手 - 打包工具
echo ==============================
echo.

:: 激活 conda 环境
call conda activate pytorch

:: 安装依赖
echo [1/3] 安装依赖...
pip install -r requirements.txt pyinstaller -q

:: 打包
echo [2/3] 打包中 (可能需要2-5分钟)...
pyinstaller --noconfirm --onedir --windowed ^
  --name "面试助手" ^
  --add-data "ui;ui" ^
  --add-data "config.py;." ^
  --add-data "engine.py;." ^
  --hidden-import sounddevice ^
  --hidden-import numpy ^
  --hidden-import requests ^
  --hidden-import faster_whisper ^
  main.py

echo [3/3] 完成!
echo.
echo 输出目录: dist\面试助手\
echo 压缩 dist\面试助手 文件夹即可分发给其他人
echo.
pause

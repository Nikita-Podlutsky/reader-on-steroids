@echo off
chcp 65001 > nul
title KOTODEX RESEARCH CORE

echo ===================================================
echo        ЗАПУСК СИСТЕМЫ KOTODEX (SOTA CORE)
echo ===================================================
echo.
echo [1/3] Проверка окружения...
if not exist ".venv\Scripts\python.exe" (
    echo ОШИБКА: Папка .venv не найдена! Распакуйте архив полностью.
    pause
    exit
)

echo [2/3] Запуск локального сервера...
echo.
echo      Не закрывайте это черное окно, пока работаете с картой!
echo      Если браузер не открылся, перейдите по ссылке:
echo      http://localhost:8000
echo.


start "" "http://localhost:8000"

REM Запускаем uvicorn через питон из виртуального окружения

.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info

pause
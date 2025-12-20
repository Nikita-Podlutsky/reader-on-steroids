@echo off
chcp 65001 > nul
title KOTODEX RESEARCH CORE - PRODUCTION BUILD

echo ===================================================
echo        ЗАПУСК СИСТЕМЫ KOTODEX (PRODUCTION)
echo ===================================================
echo.

echo [1/5] Проверка окружения...
if not exist ".venv\Scripts\python.exe" (
    echo ❌ ОШИБКА: Папка .venv не найдена! Распакуйте архив полностью.
    pause
    exit
)

if not exist "frontend\package.json" (
    echo ❌ ОШИБКА: Папка frontend не найдена!
    pause
    exit
)
echo ✓ Окружение проверено

echo.
echo [2/5] Проверка фронтенда...

REM Проверяем наличие собранных файлов
if exist "app\static\index.html" (
    if exist "app\static\assets" (
        echo ✓ Фронтенд уже собран, пропускаем сборку
        echo    Для пересборки удалите папку app\static
        goto :skip_build
    )
)

echo ⚠️  Собранный фронтенд не найден. Начинаем сборку...

where npm >nul 2>&1
if errorlevel 1 (
    echo ❌ ОШИБКА: npm не найден!
    echo    Установите Node.js с https://nodejs.org/
    pause
    exit
)
echo ✓ Node.js и npm найдены

echo.
echo [3/5] Установка зависимостей фронтенда...
cd frontend
call npm install
if errorlevel 1 (
    echo ❌ Ошибка установки зависимостей!
    cd ..
    pause
    exit
)
echo ✓ Зависимости установлены

echo.
echo [4/5] Сборка фронтенда (production)...
call npm run build
if errorlevel 1 (
    echo ❌ Ошибка сборки фронтенда!
    cd ..
    pause
    exit
)
echo ✓ Фронтенд собран

echo.
echo [5/5] Копирование файлов в app\static...
if exist "..\app\static" (
    rmdir /S /Q "..\app\static" 2>nul
)
xcopy /E /I /Y dist ..\app\static >nul
if errorlevel 1 (
    echo ❌ Ошибка копирования файлов!
    cd ..
    pause
    exit
)
cd ..
echo ✓ Файлы скопированы

:skip_build
echo.
echo ===================================================
echo        ЗАПУСК СЕРВЕРА
echo ===================================================
echo.
echo 🚀 Запуск uvicorn...
echo    Сервер будет доступен по адресу: 
echo.

REM Запускаем браузер ДО старта сервера (с задержкой)
start /B cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000/"

REM Запускаем сервер
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info

echo.
echo ⚠️  Сервер остановлен
pause
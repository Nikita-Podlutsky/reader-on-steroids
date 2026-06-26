import torch
from pathlib import Path
import shutil

import config as CONFIG

def save_checkpoint(model, optimizer, scheduler, scaler, state: dict, path: Path):
    """
    Сохраняет чекпоинт, включая LoRA-адаптеры и состояние обучения.
    Сохраняет в директорию.
    """
    # Создаем временную директорию
    temp_path = path.parent / f"{path.name}.tmp"
    temp_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if CONFIG.USE_QLORA_QUERY:
            model.query_encoder.transformer.save_pretrained(str(temp_path / "query_encoder_lora"))

        if CONFIG.USE_LORA_DOC:
            model.document_encoder.longformer.save_pretrained(str(temp_path / "document_encoder_lora"))

        torch.save(model.state_dict(), temp_path / "model_state.pt")
        
        torch.save(state, temp_path / "training_state.pt")
        
        if path.exists():
            shutil.rmtree(path)
        temp_path.rename(path)
        
        print(f"✓ Чекпоинт сохранен в директорию: {path}")

    except Exception as e:
        print(f"❌ Ошибка при сохранении чекпоинта: {e}")
        if temp_path.exists():
            shutil.rmtree(temp_path)


def load_checkpoint(model, optimizer, scaler, scheduler, path: Path):
    """
    Загружает чекпоинт, включая LoRA-адаптеры.
    """
    if not path.is_dir():
        print(f"INFO: Директория чекпоинта не найдена: {path}")
        return None
        
    try:
        print(f"INFO: Загрузка чекпоинта из директории {path}...")

        query_lora_path = path / "query_encoder_lora"
        if CONFIG.USE_QLORA_QUERY and query_lora_path.is_dir():
            model.query_encoder.transformer.load_adapter(str(query_lora_path), adapter_name="default")
            print("INFO: -> LoRA адаптеры для QueryEncoder загружены.")

        doc_lora_path = path / "document_encoder_lora"
        if CONFIG.USE_LORA_DOC and doc_lora_path.is_dir():
            model.document_encoder.longformer.load_adapter(str(doc_lora_path), adapter_name="default")
            print("INFO: -> LoRA адаптеры для DocumentEncoder загружены.")
        

        model.load_state_dict(torch.load(path / "model_state.pt", map_location=CONFIG.DEVICE), strict=False)
        print("INFO: -> Остальные веса модели загружены.")

        training_state_path = path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location=CONFIG.DEVICE)
            

            if optimizer is not None and 'optimizer_state_dict' in state:
                optimizer.load_state_dict(state['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in state:
                scheduler.load_state_dict(state['scheduler_state_dict'])
            
            if scaler is not None and 'scaler_state_dict' in state and scaler.is_enabled():
                scaler.load_state_dict(state['scaler_state_dict'])
            
            print(f"✓ Чекпоинт успешно загружен! Метаданные обучения восстановлены.")
            return state
        else:
            print("WARNING: Файл training_state.pt не найден. Загружены только веса.")
            return None

    except Exception as e:
        print(f"❌ Ошибка загрузки чекпоинта: {e}. Обучение начнется с нуля.")
        return None


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    autosave_path = checkpoint_dir / CONFIG.CHECKPOINT_FILENAME_AUTOSAVE
    if autosave_path.is_dir():
        return autosave_path

    epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_epoch_")]
    if epoch_dirs:
        return max(epoch_dirs, key=lambda d: int(d.name.split('_')[-1]))
        
    return None
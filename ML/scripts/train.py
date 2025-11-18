# ==============================================================================
# 0. ИМПОРТЫ
# ==============================================================================

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# ==============================================================================
# 1. ИМПОРТЫ КОМПОНЕНТОВ ПРОЕКТА
# ==============================================================================
from config import CONFIG
from models import UniversalScorer
from dataset import collate_for_hierarchical
from checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
from dataset import HierarchicalTripletDataset2 as Dataset
# ==============================================================================
# 2. ФУНКЦИЯ ОБУЧЕНИЯ ОДНОЙ ЭПОХИ
# ==============================================================================
def train_epoch(model, dataloader, loss_fn, optimizer, scaler, scheduler, epoch, best_loss, start_step=0):
    model.train()
    total_loss = 0.0
    data_iter = iter(dataloader)

    if start_step > 0:
        print(f"INFO: Возобновление с шага {start_step}. Пропускаем пройденные батчи...")
        for _ in range(start_step):
            try:
                next(data_iter)
            except StopIteration:
                start_step = 0
                break
    
    pbar = tqdm(data_iter, total=len(dataloader), initial=start_step, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar, start=start_step):
        try:
            for key in batch:
                if isinstance(batch[key], dict):
                    batch[key] = {k: v.to(CONFIG.DEVICE, non_blocking=True) for k, v in batch[key].items()}
            
            with autocast(device_type=CONFIG.DEVICE, dtype=torch.bfloat16 if CONFIG.USE_BF16 else torch.float16):
                anchor_vec, positive_vec, negative_vec = model(batch)
                loss = loss_fn(anchor_vec, positive_vec, negative_vec)
                loss = loss / CONFIG.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * CONFIG.ACCUMULATION_STEPS
            total_loss += current_loss
            with torch.no_grad():
                d_pos = F.pairwise_distance(anchor_vec, positive_vec)
                d_neg = F.pairwise_distance(anchor_vec, negative_vec)
                delta = (d_neg - d_pos).mean().item()
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{total_loss / (batch_idx - start_step + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                "Δ": delta
                # 'lr': str(CONFIG.LEARNING_RATE)
            })

            if CONFIG.AUTOSAVE_EVERY_N_STEPS > 0 and (batch_idx + 1) % CONFIG.AUTOSAVE_EVERY_N_STEPS == 0:
                state = {
                    'epoch': epoch, 'step': batch_idx + 1, 'loss': current_loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(), 'best_loss': best_loss
                }
                
                save_checkpoint(model, optimizer, scheduler, scaler, state, CONFIG.CHECKPOINT_DIR / CONFIG.CHECKPOINT_FILENAME_AUTOSAVE)
            
        except Exception as e:
            print(f"\n⚠ Ошибка на батче {batch_idx}: {e}")
            raise e
    
    avg_loss = total_loss / (len(dataloader) - start_step) if len(dataloader) > start_step else 0.0
    return avg_loss

# ==============================================================================
# 3. ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА
# ==============================================================================
def main():
    print("="*60 + "\n🚀 ЗАПУСК ОБУЧЕНИЯ\n" + "="*60)
    
    # --- Инициализация компонентов ---
    print("📂 Загрузка данных...")
    clean_meta_path = CONFIG.BASE_DATA_DIR / f"CLEAN_{CONFIG.FINAL_METADATA_FILE.name}"
    dataset = Dataset(metadata_file=str(clean_meta_path if clean_meta_path.exists() else CONFIG.FINAL_METADATA_FILE))
    dataloader = DataLoader(dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, collate_fn=collate_for_hierarchical, num_workers=4, pin_memory=True)

    print("🤖 Инициализация модели...")
    model = UniversalScorer()
    
    loss_fn = nn.TripletMarginLoss(margin=CONFIG.MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    
    num_training_steps = len(dataloader) * CONFIG.NUM_EPOCHS // CONFIG.ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CONFIG.WARMUP_STEPS, num_training_steps=num_training_steps)

    scaler = GradScaler(enabled=(CONFIG.DEVICE == 'cuda' and not CONFIG.USE_BF16))

    # ===> ЛОГИКА ВОЗОБНОВЛЕНИЯ ОБУЧЕНИЯ <===
    start_epoch, start_step, best_loss = 1, 0, float('inf')
    if CONFIG.RESUME_TRAINING:
        latest_checkpoint_path = find_latest_checkpoint(CONFIG.CHECKPOINT_DIR)
        if latest_checkpoint_path:
            state = load_checkpoint(model, optimizer, scaler, scheduler, latest_checkpoint_path)
            if state:
                start_epoch = state.get('epoch', 1)
                step_in_epoch = state.get('step', 0)

                best_loss = state.get('best_loss', float('inf'))
                
                if step_in_epoch >= len(dataloader):
                    start_epoch += 1
                    start_step = 0
                else:
                    start_step = step_in_epoch

    print("\n" + "="*60 + f"\n🚀 НАЧАЛО ОБУЧЕНИЯ С ЭПОХИ {start_epoch}, ШАГА {start_step}\n" + "="*60)
    
    for epoch in range(start_epoch, CONFIG.NUM_EPOCHS + 1):
        print(f"\n{'='*25} Эпоха {epoch}/{CONFIG.NUM_EPOCHS} {'='*25}")
        avg_loss = train_epoch(model, dataloader, loss_fn, optimizer, scaler, scheduler, epoch, best_loss, start_step=start_step)
        start_step = 0 # Сбрасываем для следующих эпох
        
        print(f"\n📉 Средний loss за эпоху {epoch}: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"⭐ Новая лучшая модель! Loss: {best_loss:.4f}")
            state = {
                'epoch': epoch, 'step': len(dataloader), 'loss': avg_loss, 'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict()
            }
            save_checkpoint(model, optimizer, scheduler, scaler, state, CONFIG.CHECKPOINT_DIR / CONFIG.CHECKPOINT_FILENAME_BEST)
            
        state = {
            'epoch': epoch, 'step': len(dataloader), 'loss': avg_loss, 'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict()
        }
        save_checkpoint(model, optimizer, scheduler, scaler, state, CONFIG.CHECKPOINT_DIR / CONFIG.CHECKPOINT_FILENAME_EPOCH.format(epoch=epoch))

    print("\n" + "="*60 + "\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО\n" + "="*60)

if __name__ == "__main__":
    main()
# ==============================================================================
#
#                   Скрипт Валидации и "Чистки" Метаданных
#
# ==============================================================================
#
#   Что делает:
#   1. Загружает ваш существующий `metadata_for_hierarchical.json`.
#   2. Пробегается по каждому триплету.
#   3. Проверяет, существуют ли на диске все три файла (.pt), на которые
#      ссылается триплет.
#   4. Создает НОВЫЙ, "чистый" файл метаданных, содержащий только 100% валидные
#      записи.
#
#   Запуск:
#   > python validate_metadata.py
#
#   Это займет всего пару минут, а не часы.
#
# ==============================================================================

import json
from pathlib import Path
from tqdm import tqdm

# --- Импортируем наш единый конфиг, чтобы использовать правильные пути ---
from config import CONFIG

def main():
    # Определяем пути к "грязному" и "чистому" файлам
    source_metadata_path = CONFIG.FINAL_METADATA_FILE
    # Создадим новое имя для чистого файла, чтобы не затереть старый на всякий случай
    clean_metadata_path = source_metadata_path.parent / f"CLEAN_{source_metadata_path.name}"

    print("="*60)
    print("🚀 ЗАПУСК ВАЛИДАЦИИ МЕТАДАННЫХ")
    print(f"Исходный файл: {source_metadata_path}")
    print(f"Выходной файл:  {clean_metadata_path}")
    print("="*60)

    try:
        with open(source_metadata_path, 'r', encoding='utf-8') as f:
            all_triplets = json.load(f)
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Исходный файл метаданных не найден: {source_metadata_path}")
        print("Пожалуйста, сначала запустите prepare_data.py, чтобы его создать.")
        return

    print(f"Найдено {len(all_triplets):,} триплетов в исходном файле. Начинаем проверку...")

    valid_triplets = []
    
    # tqdm будет показывать прогресс
    for triplet in tqdm(all_triplets, desc="Проверка путей"):
        # Получаем пути из каждого триплета
        anchor_path = Path(triplet.get('anchor_path'))
        positive_path = Path(triplet.get('positive_path'))
        negative_path = Path(triplet.get('negative_path'))

        # Проверяем, что все три файла существуют
        if anchor_path.exists() and positive_path.exists() and negative_path.exists():
            valid_triplets.append(triplet)

    print("\n" + "="*60)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print(f"Найдено валидных триплетов: {len(valid_triplets):,} из {len(all_triplets):,}")
    
    # Считаем, сколько "битых" записей мы отфильтровали
    filtered_count = len(all_triplets) - len(valid_triplets)
    if filtered_count > 0:
        print(f"Отфильтровано 'битых' записей: {filtered_count}")
    else:
        print("Все записи в исходном файле валидны. Отлично!")

    # Сохраняем "чистый" список в новый файл
    with open(clean_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(valid_triplets, f, indent=2, ensure_ascii=False)
        
    print(f"\n✓ 'Чистый' файл метаданных сохранен: {clean_metadata_path}")
    print("Теперь используйте его для обучения.")
    print("="*60)


if __name__ == "__main__":
    main()
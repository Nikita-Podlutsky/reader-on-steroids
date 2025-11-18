# #!/usr/bin/env python3

# import os, sys
# from llm import LLM   # твой rotate-клиент из прошлого шага

# SUMMARIZER = """Ты — рецензент научных статей. 
# Прочти текст и выдай 4-5 коротких bullet (≤ 15 слов каждый):
# - Ключевая задача / вопрос статьи
# - Главное новшество / разница с предыдущими работами
# - Основной результат (число, метрика, качественный вывод)
# - Применение / чем полезно
# - (если есть) ограничения / подводные камни

# Если текста мало, просто напиши «короткая заметка»."""

# llm = LLM()

# def summarize(text: str, max_tokens:700) -> str:
#     prompt = [{"role": "system", "content": SUMMARIZER},
#               {"role": "user",   "content": text[:12_000]}]  # хватает для большинства PDF
#     return llm.chat(prompt, temperature=0.2, max_tokens=max_tokens)

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("usage: python summarize.py paper.txt")
#         sys.exit(1)
#     with open(sys.argv[1], encoding="utf-8") as f:
#         print(summarize(f.read()))


from pathlib import Path
import torch
files = list(Path("triplets_data/sentence_embeddings_pt").rglob("*.pt"))
print("mean shape:", torch.load(files[0]).shape)
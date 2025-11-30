# real_test.py
import json
import requests
from pathlib import Path
import random

URL = "http://localhost:8000/rank"
PDF_DIR = Path("arxiv_pdfs")

# берём первые 5 pdf
temp = list(PDF_DIR.glob("*.pdf"))

random.shuffle(temp)
pdf_files = temp[:5]
print(pdf_files)
if not pdf_files:
    exit("Папка arxiv_pdfs пуста")

# придумали запрос
query = "attention mechanism in transformer models"

# формируем список документов для API
documents = [{"path": str(p.name)} for p in pdf_files]

# отправляем
resp = requests.post(
    URL,
    json={"query": query, "documents": documents},
    timeout=60
)
resp.raise_for_status()

scores = resp.json()["scores"]

# выводим только пары «файл → счёт»
for pdf, score in zip(pdf_files, scores):
    print(f"{pdf.name:<40}  {score:.3f}")
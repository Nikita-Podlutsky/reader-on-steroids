# app/arxiv_loader.py

import os
import re
import requests
import fitz  # PyMuPDF
import arxiv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict
from tqdm import tqdm

# Путь для кэша PDF
CACHE_DIR = Path(__file__).parent.parent / "arxiv_pdfs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class ArxivLoader:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        
        # Настройка клиента ArXiv (оставляем как было для стабильности)
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )

        # ОПТИМИЗАЦИЯ 1: Сессия с пулом соединений и повторными попытками
        # Это ускоряет скачивание множества файлов подряд
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=max_workers, pool_maxsize=max_workers)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        # ОПТИМИЗАЦИЯ 2: Компиляция регулярных выражений один раз
        self.re_whitespace = re.compile(r'\s+')
        self.re_links = re.compile(r'http\S+')
        self.re_pagination = re.compile(r'\b\d+\s+\b')

    def clean_text(self, text: str) -> str:
        """Промышленная очистка текста"""
        if not text: 
            return ""
        # Используем скомпилированные regex
        text = self.re_whitespace.sub(' ', text).strip()
        text = self.re_links.sub('', text)
        text = self.re_pagination.sub('', text)
        return text

    def _process_paper(self, paper_data: dict) -> dict:
        """
        Единая функция (unit of work) для потока:
        Скачать -> (если надо) -> Вытащить текст -> Вернуть результат
        """
        paper_id = paper_data['id']
        pdf_url = paper_data['pdf_url']
        file_path = CACHE_DIR / f"{paper_id}.pdf"

        # 1. Скачивание (если нет в кэше)
        if not file_path.exists():
            try:
                # Используем сессию
                with self.session.get(pdf_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"Failed to download {paper_id}: {e}")
                # Если не удалось скачать, возвращаем то, что есть (заголовок + абстракт)
                paper_data['full_text'] = f"{paper_data['title']}. {paper_data['abstract']}"
                return paper_data

        # 2. Парсинг (сразу же, пока файл "горячий" в кэше ОС)
        full_text = ""
        try:
            # fitz очень быстр, чтение с диска здесь не узкое место
            with fitz.open(file_path) as doc:
                # Сразу генератором собираем текст
                full_text = " ".join(page.get_text() for page in doc)
            
            full_text = self.clean_text(full_text)
        except Exception as e:
            print(f"Failed to parse PDF {paper_id}: {e}")

        # Fallback если текст пустой или битый
        if len(full_text) < 500:
            full_text = f"{paper_data['title']}. {paper_data['abstract']}"

        paper_data['full_text'] = full_text
        return paper_data

    def search_and_load(self, query: str, max_results=30) -> List[Dict]:
        """
        Главный метод: Ищет и запускает конвейер обработки параллельно.
        """
        print(f"🔍 Searching ArXiv for: '{query}' (Limit: {max_results})...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # Список задач
        futures = []
        final_papers = []

        # ОПТИМИЗАЦИЯ 3: Конвейерная обработка
        # Мы не ждем, пока скачаются ВСЕ метаданные. Мы добавляем задачи в пул по мере прихода данных.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Генератор результатов от ArXiv
            results_gen = self.client.results(search)
            
            try:
                for r in results_gen:
                    short_id = r.get_short_id().split('v')[0]
                    paper_meta = {
                        "id": short_id,
                        "title": r.title.replace('\n', ' '),
                        "abstract": r.summary.replace('\n', ' '),
                        "pdf_url": r.pdf_url,
                        "published": str(r.published.date())
                    }
                    
                    # Сразу кидаем в работу
                    futures.append(executor.submit(self._process_paper, paper_meta))
            except Exception as e:
                print(f"Error fetching metadata: {e}")

            print(f"Found {len(futures)} papers. Processing (Download + Parse)...")

            # Сбор результатов по мере готовности (as_completed)
            # Это позволяет видеть прогрессбар "живым"
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    final_papers.append(result)
                except Exception as e:
                    print(f"Task failed: {e}")

        print(f"✅ Processed {len(final_papers)} papers ready for analysis.")
        return final_papers
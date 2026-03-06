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


CACHE_DIR = Path(__file__).parent.parent / "arxiv_pdfs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class ArxivLoader:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        
        
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )

        
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=max_workers, pool_maxsize=max_workers)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        
        self.re_whitespace = re.compile(r'\s+')
        self.re_links = re.compile(r'http\S+')
        self.re_pagination = re.compile(r'\b\d+\s+\b')

    def clean_text(self, text: str) -> str:
        """Промышленная очистка текста"""
        if not text: 
            return ""
        
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

        
        if not file_path.exists():
            try:
                
                with self.session.get(pdf_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"Failed to download {paper_id}: {e}")
                
                paper_data['full_text'] = f"{paper_data['title']}. {paper_data['abstract']}"
                return paper_data

        
        full_text = ""
        try:
            
            with fitz.open(file_path) as doc:
                
                page_texts = []
                for page_num, page in enumerate(doc):
                    try:
                        
                        page_text = page.get_text()
                        if page_text:
                            page_texts.append(page_text)
                    except (RuntimeError, ValueError, AttributeError) as page_error:
                        
                        error_str = str(page_error).lower()
                        if 'extgstate' in error_str or 'syntax error' in error_str or 'resource' in error_str:
                            
                            try:
                                page_text = page.get_text("dict")
                                if page_text and 'blocks' in page_text:
                                    text_parts = []
                                    for block in page_text['blocks']:
                                        if 'lines' in block:
                                            for line in block['lines']:
                                                if 'spans' in line:
                                                    for span in line['spans']:
                                                        if 'text' in span:
                                                            text_parts.append(span['text'])
                                    if text_parts:
                                        page_texts.append(' '.join(text_parts))
                                        continue
                            except Exception:
                                pass
                            
                            try:
                                
                                page_text = page.get_text("rawdict")
                                if page_text and 'blocks' in page_text:
                                    text_parts = []
                                    for block in page_text['blocks']:
                                        if isinstance(block, dict) and 'lines' in block:
                                            for line in block['lines']:
                                                if isinstance(line, dict) and 'spans' in line:
                                                    for span in line['spans']:
                                                        if isinstance(span, dict) and 'text' in span:
                                                            text_parts.append(span['text'])
                                    if text_parts:
                                        page_texts.append(' '.join(text_parts))
                                        continue
                            except Exception:
                                pass
                            
                            
                            print(f"Warning: Skipping page {page_num+1}/{len(doc)} in {paper_id} due to PDF parsing error (ExtGState/syntax)")
                        else:
                            
                            print(f"Warning: Skipping page {page_num+1}/{len(doc)} in {paper_id}: {page_error}")
                    except Exception as page_error:
                        
                        print(f"Warning: Error on page {page_num+1}/{len(doc)} in {paper_id}: {page_error}")
                        continue
                
                full_text = " ".join(page_texts)
            
            full_text = self.clean_text(full_text)
        except Exception as e:
            print(f"Failed to parse PDF {paper_id}: {e}")
            
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    full_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                    full_text = self.clean_text(full_text)
                    print(f"Successfully parsed {paper_id} using pypdf fallback")
            except ImportError:
                
                pass
            except Exception as fallback_error:
                print(f"Fallback parsing also failed for {paper_id}: {fallback_error}")

        
        if len(full_text) < 500:
            if full_text:
                
                full_text = f"{paper_data['title']}. {paper_data['abstract']}. {full_text[:1000]}"
            else:
                
                full_text = f"{paper_data['title']}. {paper_data['abstract']}"
                print(f"Warning: {paper_id} - using abstract only (PDF parsing failed or empty)")

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

        
        futures = []
        final_papers = []

        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            
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
                    
                    
                    futures.append(executor.submit(self._process_paper, paper_meta))
            except Exception as e:
                print(f"Error fetching metadata: {e}")

            print(f"Found {len(futures)} papers. Processing (Download + Parse)...")

            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    final_papers.append(result)
                except Exception as e:
                    print(f"Task failed: {e}")

        print(f"✅ Processed {len(final_papers)} papers ready for analysis.")
        return final_papers
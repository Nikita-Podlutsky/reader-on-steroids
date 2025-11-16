#!/usr/bin/env python3
"""
Минимальный пример использования ScholarMap API
Запуск: python quick_test.py
"""

import requests
import json

API_URL = "http://localhost:8001"

# Минимальный набор документов для теста
DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and decoder. The best performing models also connect the encoder and decoder through an attention mechanism.",
        "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
        "year": 2017,
        "keywords": ["transformers", "attention", "neural networks"]
    },
    {
        "id": "doc_2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations.",
        "authors": ["Devlin, J.", "Chang, M."],
        "year": 2018,
        "keywords": ["BERT", "pre-training", "transformers"]
    },
    {
        "id": "doc_3",
        "title": "Language Models are Few-Shot Learners",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
        "authors": ["Brown, T.", "Mann, B."],
        "year": 2020,
        "keywords": ["GPT-3", "few-shot learning", "language models"]
    },
    {
        "id": "doc_4",
        "title": "Quantum Computing: A Gentle Introduction",
        "abstract": "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers.",
        "authors": ["Nielsen, M.", "Chuang, I."],
        "year": 2019,
        "keywords": ["quantum computing", "quantum mechanics", "qubits"]
    },
    {
        "id": "doc_5",
        "title": "Variational Quantum Algorithms",
        "abstract": "Variational quantum algorithms are a class of quantum algorithms that use a classical optimizer to train a parameterized quantum circuit. These algorithms have shown promise for near-term quantum computers with applications in chemistry, optimization, and machine learning.",
        "authors": ["Cerezo, M.", "Arrasmith, A."],
        "year": 2021,
        "keywords": ["quantum algorithms", "variational circuits", "QAOA"]
    },
    {
        "id": "doc_6",
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "abstract": "Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results.",
        "authors": ["Liu, Y.", "Ott, M."],
        "year": 2019,
        "keywords": ["RoBERTa", "BERT", "optimization"]
    },
]

def main():
    print("🚀 ScholarMap API Quick Test")
    print("="*60)
    
    # 1. Проверка здоровья API
    print("\n1️⃣  Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API is healthy")
            print(f"   Device: {health['device']}")
            print(f"   Model: {health['model']}")
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ API is not accessible: {e}")
        print(f"   💡 Make sure to start the API first:")
        print(f"      python scholarmap_api.py")
        return
    
    # 2. Отправка документов на анализ
    print(f"\n2️⃣  Analyzing {len(DOCUMENTS)} documents...")
    request_data = {
        "documents": DOCUMENTS,
        "min_similarity": 0.75,
        "max_topics": 10
    }
    
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json=request_data,
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"   ❌ Request failed: {response.status_code}")
            print(response.text)
            return
        
        result = response.json()
        print("   ✅ Analysis completed!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 3. Печать результатов
    print("\n3️⃣  Results:")
    print("="*60)
    
    # Темы
    print(f"\n📚 Found {len(result['topics'])} topics:")
    for topic in result['topics']:
        print(f"\n   • {topic['name']} ({topic['doc_count']} documents)")
        print(f"     Keywords: {', '.join(topic['keywords'][:4])}")
        print(f"     Color: RGB{tuple(topic['color_rgb'])}")
    
    # Статистика
    stats = result['statistics']
    print(f"\n📊 Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   🔴 Identical pairs: {stats['identical_pairs']}")
    print(f"   🟠 Very similar pairs: {stats['very_similar_pairs']}")
    print(f"   🟡 Similar pairs: {stats['similar_pairs']}")
    print(f"   Average connections per doc: {stats['avg_connections_per_doc']:.2f}")
    
    # Кластеры
    print(f"\n🗂️  Clusters:")
    for cluster in result['clusters']:
        print(f"\n   Cluster: {cluster['name']}")
        print(f"   Size: {cluster['size']} documents")
        print(f"   Diversity: {cluster['diversity_score']:.3f}")
        print(f"   Documents: {', '.join(cluster['doc_ids'])}")
    
    # Примеры похожих пар
    print(f"\n🔗 Sample similar pairs:")
    
    edges_by_level = {}
    for edge in result['edges']:
        level = edge['level']
        if level not in edges_by_level:
            edges_by_level[level] = []
        edges_by_level[level].append(edge)
    
    for level in ['identical', 'very_similar', 'similar']:
        if level in edges_by_level and edges_by_level[level]:
            symbol = {'identical': '🔴', 'very_similar': '🟠', 'similar': '🟡'}[level]
            print(f"\n   {symbol} {level.upper()}:")
            for edge in edges_by_level[level][:3]:
                src_title = next(n['title'] for n in result['nodes'] if n['id'] == edge['source'])
                tgt_title = next(n['title'] for n in result['nodes'] if n['id'] == edge['target'])
                print(f"      {edge['source']} ↔ {edge['target']} (similarity: {edge['similarity']:.4f})")
                print(f"         • {src_title[:50]}...")
                print(f"         • {tgt_title[:50]}...")
    
    # 4. Сохранение результата
    print("\n4️⃣  Saving results...")
    output_file = "scholarmap_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved to {output_file}")
    
    # 5. Инструкции по использованию
    print("\n" + "="*60)
    print("🎉 Test completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Check the result in scholarmap_result.json")
    print("   2. Use nodes and edges to build a graph visualization")
    print("   3. Run full test: python test_both_apis.py")
    print("   4. Open Swagger UI: http://localhost:8001/")
    print("="*60)

if __name__ == "__main__":
    main()
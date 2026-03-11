import ollama
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class DirectPromptingExtractor:
    def __init__(self, model_name="gemma3:1b", temperature=0.1):
        self.model_name = model_name
        # Thêm num_predict để ép nó chỉ sinh ra tối đa 100 token
        self.options = {
            "temperature": temperature,
            "num_predict": 100, 
            "top_p": 0.9
        }

    def _process_batch(self, batch):
        """Worker function to handle a single batch for parallel execution."""
        reviews_text = "\n".join([f"Review: {text}" for text in batch])
        
        # Prompt nghiêm ngặt hơn + Có ví dụ mẫu (Few-shot)
        prompt = (
            "You are an expert data extractor. Your ONLY job is to extract 1-2 core aspect keywords (like 'Engine', 'Price', 'Comfort') from car reviews.\n\n"
            "STRICT RULES:\n"
            "1. DO NOT write full sentences.\n"
            "2. DO NOT provide explanations or introductory text.\n"
            "3. Output format must be a SINGLE comma-separated list of keywords.\n\n"
            "EXAMPLE:\n"
            "Review: The seats are very uncomfortable for long trips, but the gas mileage is great.\n"
            "Output: Comfort, Fuel Economy\n\n"
            "YOUR TURN:\n"
            f"{reviews_text}\n\n"
            "Output:"
        )
        
        try:
            res = ollama.generate(model=self.model_name, prompt=prompt, options=self.options)
            response_text = res['response'].strip()
            
            # Loại bỏ chữ "Output:" nếu AI lỡ in lại
            if "Output:" in response_text:
                response_text = response_text.split("Output:")[-1]
                
            raw_kws = [kw.strip() for kw in response_text.split(',')]
            
            # Chỉ giữ lại các từ khóa có độ dài từ 1 đến 3 từ (loại bỏ các câu văn dài)
            cleaned_kws = [kw for kw in raw_kws if 0 < len(kw.split()) <= 3]
            
            return cleaned_kws
        except Exception as e:
            print(f"Error processing batch: {e}")
            return []

    def extract_initial_aspects(self, texts, batch_size=5, max_workers=4):
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        raw_aspects = []
        
        print(f"Extracting aspects via LLM (Parallel mode, {max_workers} workers)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the worker function to the list of batches
            results = list(tqdm(executor.map(self._process_batch, batches), total=len(batches), desc="Processing Batches"))
            
        for res in results:
            raw_aspects.extend(res)
            
        # Chuẩn hóa viết hoa chữ cái đầu (VD: 'engine' và 'Engine' sẽ gộp thành 1)
        unique_aspects = list(set([aspect.title() for aspect in raw_aspects if aspect]))
        return unique_aspects

    def refine_topics(self, raw_aspects, n_final_topics=5):
        aspects_str = ", ".join(raw_aspects)
        # Ép chặt số lượng topic cuối cùng và cung cấp mẫu
        prompt = (
            f"Group these overlapping keywords into EXACTLY {n_final_topics} broad topic categories for cars.\n"
            "STRICT RULES:\n"
            "1. Return ONLY the category names separated by commas.\n"
            "2. No introductory text or bullet points.\n"
            "Example: Engine Performance, Interior Comfort, Customer Service, Pricing, Reliability\n\n"
            f"Keywords to group: {aspects_str}\n\n"
            "Output:"
        )
        
        res = ollama.generate(model=self.model_name, prompt=prompt, options=self.options)
        response_text = res['response'].strip()
        
        if "Output:" in response_text:
            response_text = response_text.split("Output:")[-1]
            
        # Cắt bỏ dấu chấm ở cuối nếu có
        return [t.strip().strip('.') for t in response_text.split(',')][:n_final_topics]

class EmbeddingClusteringExtractor:
    def __init__(self, n_clusters=5, llm_model="gemma3:1b", embed_model='all-MiniLM-L6-v2'):
        self.n_clusters = n_clusters
        self.llm_model = llm_model
        self.embedder = SentenceTransformer(embed_model)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = None

    def fit_predict(self, texts):
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        self.labels = self.kmeans.fit_predict(embeddings)
        return self.labels

    def _interpret_single_cluster(self, args):
        """Worker function to interpret a single cluster for parallel execution."""
        cluster_id, sample_texts = args
        samples_str = "\n".join([f"- {t}" for t in sample_texts])
        prompt = (
            "Analyze these car reviews and provide a single short category name (max 3 words) "
            "that summarizes their common topic. Output ONLY the name:\n\n"
            f"{samples_str}"
        )
        res = ollama.generate(model=self.llm_model, prompt=prompt, options={'temperature': 0.1})
        return f"Cluster {cluster_id}", res['response'].strip()

    def interpret_clusters(self, texts, max_workers=4):
        df = pd.DataFrame({'text': texts, 'cluster': self.labels})
        cluster_summaries = {}
        
        # Prepare the arguments for the thread pool
        cluster_tasks = []
        for i in range(self.n_clusters):
            sample_texts = df[df['cluster'] == i]['text'].head(5).tolist()
            cluster_tasks.append((i, sample_texts))
            
        print(f"Interpreting {self.n_clusters} clusters via LLM (Parallel mode, {max_workers} workers)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self._interpret_single_cluster, cluster_tasks), total=len(cluster_tasks), desc="Naming Clusters"))
            
        for cluster_id, name in results:
            cluster_summaries[cluster_id] = name
            
        return cluster_summaries
import re
import json
import ollama
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    for candidate in [
        text,
        re.sub(r"```(?:json)?|```", "", text).strip(),
    ]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _llm_options(temperature: float = 0.0) -> dict:
    return {
        "temperature": temperature,
        "top_p": 0.85,
        "num_ctx": 512,
        "num_predict": 64,      # cluster names are short
    }


# ── DirectPromptingExtractor (unchanged) ──────────────────────────────────────

class DirectPromptingExtractor:
    _EXTRACT_PROMPT = (
        'Extract 1-3 car aspect keywords from these reviews.\n'
        'Reply with ONLY this JSON, nothing else:\n'
        '{{"keywords": ["Keyword1", "Keyword2"]}}\n\n'
        'Reviews:\n{reviews}'
    )
    _REFINE_PROMPT = (
        'Group these keywords into exactly {n} broad car-review topics.\n'
        'Reply with ONLY a comma-separated list, e.g.:\n'
        'Engine Performance, Reliability, Interior Comfort, Pricing, Customer Service\n\n'
        'Keywords: {keywords}\n\nTopics:'
    )

    def __init__(self, model_name: str = "llama3.2:3b", temperature: float = 0.0):
        self.model_name = model_name
        self.options = _llm_options(temperature)

    def _call_llm(self, prompt: str) -> str:
        return ollama.generate(model=self.model_name, prompt=prompt,
                               options=self.options)["response"].strip()

    def _process_batch(self, batch: list[str]) -> list[str]:
        reviews = "\n".join(f"- {t[:120]}" for t in batch)
        try:
            raw  = self._call_llm(self._EXTRACT_PROMPT.format(reviews=reviews))
            data = _extract_json(raw)
            if data is None:
                return []
            return [kw.strip().title() for kw in data.get("keywords", [])
                    if isinstance(kw, str) and kw.strip()]
        except Exception as e:
            print(f"[batch error] {e}")
            return []

    def extract_initial_aspects(self, texts: list[str],
                                 batch_size: int = 5,
                                 max_workers: int = 4) -> list[str]:
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        print(f"Extracting aspects ({len(batches)} batches, {max_workers} workers)…")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(tqdm(pool.map(self._process_batch, batches),
                                total=len(batches)))
        flat = [kw for r in results for kw in r]
        return [kw for kw, _ in Counter(flat).most_common()]

    def refine_topics(self, raw_aspects: list[str],
                      n_final_topics: int = 4) -> list[str]:
        keywords_str = ", ".join(raw_aspects[:50])
        raw = self._call_llm(
            self._REFINE_PROMPT.format(n=n_final_topics, keywords=keywords_str)
        )
        raw = re.split(r"Topics:", raw, flags=re.IGNORECASE)[-1].strip()
        return [t.strip().strip(".") for t in raw.split(",") if t.strip()][:n_final_topics]


# ── EmbeddingClusteringExtractor ──────────────────────────────────────────────

class EmbeddingClusteringExtractor:
    """
    Pipeline
    --------
    1. Encode reviews → dense vectors  (SentenceTransformer)
    2. Cluster vectors                 (KMeans)
    3. Name each cluster               (Ollama LLM, parallel)

    Parameters
    ----------
    n_clusters   : number of topic clusters
    embed_model  : any SentenceTransformer-compatible model ID
    llm_model    : Ollama model used for cluster naming
    sample_per_cluster : reviews fed to the LLM per cluster (keep small → fast)
    max_workers  : parallel threads for LLM naming step
    """

    _NAME_PROMPT = (
        "These are car reviews from the same group:\n{samples}\n\n"
        "Give ONE short topic label (2-4 words) for this group.\n"
        "Reply with ONLY the label, nothing else."
    )

    def __init__(
        self,
        n_clusters: int = 4,
        embed_model: str = "all-MiniLM-L6-v2",   # 80 MB, very fast
        llm_model: str = "llama3.2:3b",
        sample_per_cluster: int = 5,
        max_workers: int = 4,
    ):
        self.n_clusters          = n_clusters
        self.sample_per_cluster  = sample_per_cluster
        self.max_workers         = max_workers
        self.llm_model           = llm_model
        self.llm_options         = _llm_options(temperature=0.0)

        print(f"Loading embedding model '{embed_model}'…")
        self.embedder = SentenceTransformer(embed_model)
        self.kmeans   = KMeans(n_clusters=n_clusters, random_state=42,
                               n_init="auto")   # n_init="auto" → faster in sklearn ≥1.4

        # filled after fit()
        self.labels_     : np.ndarray | None = None
        self.embeddings_ : np.ndarray | None = None
        self.cluster_names_: dict[int, str]  = {}

    # ── Step 1 : embed ─────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return normalize(vecs)   # cosine-friendly, improves KMeans quality

    # ── Step 2 : cluster ───────────────────────────────────────────────────────

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        print(f"Clustering {len(embeddings)} embeddings into {self.n_clusters} clusters…")
        return self.kmeans.fit_predict(embeddings)

    # ── Step 3 : name clusters (parallel LLM calls) ────────────────────────────

    def _name_one_cluster(self, args: tuple[int, list[str]]) -> tuple[int, str]:
        cluster_id, samples = args
        sample_block = "\n".join(f"- {t[:120]}" for t in samples)
        prompt = self._NAME_PROMPT.format(samples=sample_block)
        try:
            res = ollama.generate(model=self.llm_model, prompt=prompt,
                                  options=self.llm_options)
            return cluster_id, res["response"].strip().strip(".")
        except Exception as e:
            print(f"[cluster {cluster_id} naming error] {e}")
            return cluster_id, f"Cluster {cluster_id}"

    def _name_clusters(self, texts: list[str]) -> dict[int, str]:
        df = pd.DataFrame({"text": texts, "cluster": self.labels_})

        tasks = [
            (
                cid,
                df[df["cluster"] == cid]["text"]
                  .sample(min(self.sample_per_cluster, (df["cluster"] == cid).sum()),
                          random_state=42)
                  .tolist(),
            )
            for cid in range(self.n_clusters)
        ]

        print(f"Naming {self.n_clusters} clusters via LLM ({self.max_workers} workers)…")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(
                tqdm(pool.map(self._name_one_cluster, tasks),
                     total=self.n_clusters, desc="Naming clusters")
            )
        return dict(results)

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "EmbeddingClusteringExtractor":
        """Embed → cluster → name.  Returns self for chaining."""
        self.embeddings_ = self._embed(texts)
        self.labels_     = self._cluster(self.embeddings_)
        self.cluster_names_ = self._name_clusters(texts)
        return self

    def get_results(self, texts: list[str]) -> pd.DataFrame:
            """Return a DataFrame with columns: STT | cluster_id | topic | text."""
            if self.labels_ is None:
                raise RuntimeError("Call fit() first.")

            return pd.DataFrame({
                "STT":        range(1, len(texts) + 1),                     
                "cluster_id": self.labels_,                                 
                "topic":      [self.cluster_names_[c] for c in self.labels_],
                "text":       texts,                                         
            })

    def topic_summary(self) -> dict[str, int]:
        """{ topic_name: count } sorted by count desc."""
        if self.labels_ is None:
            raise RuntimeError("Call fit() first.")
        counts = Counter(self.labels_)
        return {
            self.cluster_names_[cid]: cnt
            for cid, cnt in counts.most_common()
        }
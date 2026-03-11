import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel

class MLTopicExtractor:
    def __init__(self, k_min=2, k_max=8):
        self.k_min = k_min
        self.k_max = k_max
        self.best_model = None
        self.optimal_k = None

    def find_optimal_k(self, X_tfidf, texts, dictionary, feature_names):
        models = []
        scores = []
        for k in range(self.k_min, self.k_max + 1):
            nmf_model = NMF(n_components=k, random_state=42, max_iter=500, init='nndsvd')
            nmf_model.fit(X_tfidf)
            models.append(nmf_model)
            
            topics = []
            for topic in nmf_model.components_:
                topics.append([feature_names[i] for i in topic.argsort()[:-11:-1]])
            
            cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
            scores.append(cm.get_coherence())
            
        optimal_idx = np.argmax(scores)
        self.optimal_k = self.k_min + optimal_idx
        self.best_model = models[optimal_idx]
        return self.optimal_k, scores[optimal_idx]

    def get_topics(self, feature_names, n_top_words=10):
        topics = {}
        for idx, topic in enumerate(self.best_model.components_):
            topics[f"Topic {idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return pd.DataFrame(topics)

    def summarize_results(self, X_tfidf, original_texts):
        # 1. Biến đổi dữ liệu sang không gian Topic
        W = self.best_model.transform(X_tfidf)
        
        # 2. Xác định Topic chủ đạo cho mỗi văn bản
        dominant_topics = np.argmax(W, axis=1) + 1 # +1 để bắt đầu từ Topic 1
        confidences = np.max(W, axis=1)
        
        summary_df = pd.DataFrame({
            'Text': original_texts,
            'Dominant_Topic': dominant_topics,
            'Confidence': confidences
        })
        
        # 3. Thống kê số lượng phần tử
        topic_counts = summary_df['Dominant_Topic'].value_counts().sort_index()
        topic_pct = (topic_counts / len(summary_df) * 100).round(2)
        
        stats = pd.DataFrame({
            'Count': topic_counts,
            'Percentage (%)': topic_pct
        })
        
        return stats, summary_df
"""
utils/topic/visualizer.py
=========================
Pure visualization layer cho Topic Extraction.
Không chứa business logic — chỉ nhận data đã xử lý và vẽ.
Dùng chung cho cả MLTopicExtractor lẫn LLMTopicExtractor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


class TopicVisualizer:

    # ── LDA Visualizations ────────────────────────────────────────────────────

    @staticmethod
    def plot_optimal_k(df: pd.DataFrame):
        """Elbow chart: perplexity + log-likelihood theo số topics K."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(df["n_topics"], df["perplexity"],     "o-", color="steelblue", linewidth=2)
        axes[1].plot(df["n_topics"], df["log_likelihood"], "s-", color="coral",     linewidth=2)
        for ax, title in zip(axes, ["Perplexity (thấp = tốt)", "Log-Likelihood (cao = tốt)"]):
            ax.set_title(title); ax.set_xlabel("Số Topics (K)"); ax.grid(alpha=0.3)
        plt.suptitle("Chọn số Topics tối ưu cho LDA", fontsize=13)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_lda_topics(lda_components, feature_names, n_topics: int, n_top_words: int):
        """Bar chart top-words cho từng LDA topic."""
        ncols  = 4
        nrows  = (n_topics + ncols - 1) // ncols
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
        axes = axes.flatten()

        for tid, vec in enumerate(lda_components):
            top_idx = vec.argsort()[-n_top_words:][::-1]
            words   = [feature_names[i] for i in top_idx]
            scores  = [vec[i]           for i in top_idx]
            axes[tid].barh(words[::-1], scores[::-1], color=colors[tid % len(colors)], alpha=0.82)
            axes[tid].set_title(f"Topic {tid}", fontsize=11, fontweight="bold")
            axes[tid].tick_params(labelsize=8); axes[tid].grid(axis="x", alpha=0.3)

        for ax in axes[n_topics:]:
            ax.axis("off")

        plt.suptitle(f"LDA — Top {n_top_words} từ / {n_topics} Topics", fontsize=14)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_topic_distribution(dominant: np.ndarray, labels: list = None):
        """Phân phối dominant topic, tuỳ chọn breakdown theo sentiment."""
        if labels is None:
            counts = pd.Series(dominant).value_counts().sort_index()
            plt.figure(figsize=(9, 4))
            plt.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
            plt.title("Phân phối Dominant Topic")
            plt.xlabel("Topic ID"); plt.ylabel("Số reviews"); plt.grid(axis="y", alpha=0.3)
            plt.tight_layout(); plt.show()
            return

        df_plot  = pd.DataFrame({"topic": dominant, "class": labels})
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        counts = pd.Series(dominant).value_counts().sort_index()
        axes[0].bar(counts.index, counts.values, color="steelblue", alpha=0.8)
        axes[0].set_title("Phân phối Dominant Topic")
        axes[0].set_xlabel("Topic ID"); axes[0].set_ylabel("Số reviews"); axes[0].grid(axis="y", alpha=0.3)

        crosstab = pd.crosstab(df_plot["topic"], df_plot["class"], normalize="columns") * 100
        crosstab.plot(kind="bar", ax=axes[1], colormap="Set2", alpha=0.85)
        axes[1].set_title("Phân phối Topic theo Sentiment (%)")
        axes[1].set_xlabel("Topic ID"); axes[1].set_ylabel("Tỷ lệ (%)")
        axes[1].legend(title="Class"); axes[1].grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout(); plt.show()

    # ── LLM Visualizations ────────────────────────────────────────────────────

    @staticmethod
    def plot_topic_frequency(df: pd.DataFrame, topic_lists: list, labels: list = None):
        """Bar chart tần suất topics + breakdown Pos/Neg."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        colors = sns.color_palette("husl", len(df))
        axes[0].barh(df["topic"][::-1], df["count"][::-1], color=colors[::-1], alpha=0.85)
        axes[0].set_title(f"Top {len(df)} Topics — LLM Direct Prompting")
        axes[0].set_xlabel("Tần suất"); axes[0].grid(axis="x", alpha=0.3)

        top10 = df["topic"].head(10).tolist()
        if labels is not None:
            pos_c = [sum(1 for tl, lb in zip(topic_lists, labels)
                        if lb == "Pos" and any(t in w.lower() for w in tl))
                     for t in top10]
            neg_c = [sum(1 for tl, lb in zip(topic_lists, labels)
                        if lb == "Neg" and any(t in w.lower() for w in tl))
                     for t in top10]
            x, w = np.arange(len(top10)), 0.4
            axes[1].bar(x - w/2, pos_c, w, label="Pos", color="steelblue", alpha=0.8)
            axes[1].bar(x + w/2, neg_c, w, label="Neg", color="coral",     alpha=0.8)
            axes[1].set_xticks(x); axes[1].set_xticklabels(top10, rotation=30, ha="right", fontsize=9)
            axes[1].set_title("Top 10 Topics theo Sentiment")
            axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].pie(df["count"][:10], labels=top10, autopct="%1.1f%%",
                        colors=sns.color_palette("pastel", 10))
            axes[1].set_title("Top 10 Topics Distribution")

        plt.suptitle("LLM Direct Prompting — Topic Frequency", fontsize=13)
        plt.tight_layout(); plt.show()

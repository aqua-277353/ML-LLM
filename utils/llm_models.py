import pandas as pd
import re
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, classification_report


class LLMSentimentAnalyzer:

    def __init__(self, model_name="llama3", temperature=0.0):

        print(f"Đang khởi tạo kết nối với Ollama model: {model_name}...")

        # Giới hạn token output để tăng tốc
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            num_predict=5
        )

        # Zero-shot prompt
        zero_template = """You are an expert data analyst evaluating car reviews.
Classify each review sentiment as 'Pos' or 'Neg'.

Return ONLY the result in this format:
1: Pos
2: Neg

Reviews:
{reviews}

Sentiments:"""

        # Few-shot prompt
        few_template = """You are an expert data analyst evaluating car reviews.
Classify each review sentiment as 'Pos' or 'Neg'.

Examples:
Review: The car broke down after 2 weeks.
Sentiment: Neg

Review: I absolutely love the fuel efficiency.
Sentiment: Pos

Now classify the following reviews.

Return format:
1: Pos
2: Neg

Reviews:
{reviews}

Sentiments:"""

        self.zero_prompt = PromptTemplate(
            input_variables=["reviews"],
            template=zero_template
        )

        self.few_prompt = PromptTemplate(
            input_variables=["reviews"],
            template=few_template
        )

        self.zero_chain = self.zero_prompt | self.llm
        self.few_chain = self.few_prompt | self.llm


    def _parse_output(self, text):

        results = re.findall(r'\b(Pos|Neg)\b', text, re.IGNORECASE)

        if results:
            return [r.capitalize() for r in results]

        return ["Unknown"]


    def _batch_reviews(self, texts):

        formatted = ""
        for i, t in enumerate(texts, 1):
            formatted += f"{i}. {t}\n"

        return formatted


    def predict_zero_shot(self, text):

        batch = self._batch_reviews([text])
        raw = self.zero_chain.invoke({"reviews": batch})
        return self._parse_output(raw)[0]


    def predict_few_shot(self, text):

        batch = self._batch_reviews([text])
        raw = self.few_chain.invoke({"reviews": batch})
        return self._parse_output(raw)[0]


    def evaluate_dataset(self, X_test, y_test, strategy="zero_shot", batch_size=8):

        y_pred = []

        print(f"Đang chạy LLM inference với chiến lược: {strategy}...")

        if strategy == "zero_shot":
            chain = self.zero_chain
        elif strategy == "few_shot":
            chain = self.few_chain
        else:
            raise ValueError("Chiến lược không hợp lệ.")

        for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):

            batch_texts = X_test[i:i + batch_size]

            formatted_reviews = self._batch_reviews(batch_texts)

            raw_response = chain.invoke({"reviews": formatted_reviews})

            preds = self._parse_output(raw_response)

            if len(preds) < len(batch_texts):
                preds += ["Unknown"] * (len(batch_texts) - len(preds))

            y_pred.extend(preds[:len(batch_texts)])

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return y_pred, acc, report
import pandas as pd
import re
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor

class LLMSentimentAnalyzer:
    def __init__(self, model_name="gemma3:1b", temperature=0.0):
        """
        Khởi tạo kết nối với model chạy local qua Ollama và chuẩn bị sẵn các chains.
        """
        print(f"Đang khởi tạo kết nối với Ollama model: {model_name}...")
        self.llm = Ollama(model=model_name, temperature=temperature)
        
        # TỐI ƯU HÓA: Khởi tạo Prompt và Chain một lần duy nhất tại đây
        self.zero_shot_chain = PromptTemplate(
            input_variables=["review"], 
            template="""You are an expert data analyst evaluating car reviews.
        Classify the sentiment of the following car review as strictly 'Pos' (Positive) or 'Neg' (Negative).
        Output ONLY the word 'Pos' or 'Neg'. Do not provide any explanations.

        Review: "{review}"
        Sentiment:"""
        ) | self.llm

        self.few_shot_chain = PromptTemplate(
            input_variables=["review"], 
            template="""You are an expert data analyst evaluating car reviews.
        Classify the sentiment of the following car review as strictly 'Pos' or 'Neg'.

        Here are some examples:
        Review: "The car broke down after 2 weeks, terrible customer service."
        Sentiment: Neg

        Review: "I absolutely love the fuel efficiency and the sleek interior design."
        Sentiment: Pos

        Now classify this review:
        Review: "{review}"
        Sentiment:"""
        ) | self.llm
        
    def _parse_output(self, text):
        match = re.search(r'\b(Pos|Neg)\b', text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        return "Unknown"

    def predict_zero_shot(self, text):
        raw_response = self.zero_shot_chain.invoke({"review": text})
        return self._parse_output(raw_response)

    def predict_few_shot(self, text):
        raw_response = self.few_shot_chain.invoke({"review": text})
        return self._parse_output(raw_response)

    def evaluate_dataset(self, X_test, y_test, strategy="zero_shot", batch_mode="sequential", max_workers=4):
        """
        Chạy dự đoán trên tập test. Hỗ trợ chạy song song qua tham số batch_mode.
        """
        y_pred = []
        print(f"Đang chạy LLM inference: Chiến lược={strategy} | Chế độ={batch_mode}...")
        
        # Xác định hàm dự đoán dựa trên chiến lược
        predict_func = self.predict_zero_shot if strategy == "zero_shot" else self.predict_few_shot

        if batch_mode == "parallel":
            # Sử dụng ThreadPoolExecutor để chạy song song
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Hàm map sẽ giữ nguyên thứ tự kết quả tương ứng với X_test
                results = list(tqdm(executor.map(predict_func, X_test), total=len(X_test), desc="Predicting (Parallel)"))
                y_pred.extend(results)
        else:
            for text in tqdm(X_test, desc="Predicting (Sequential)"):
                y_pred.append(predict_func(text))
            
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return y_pred, acc, report
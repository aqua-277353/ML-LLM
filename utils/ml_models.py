import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class SentimentModel:
    def __init__(self):
        self.available_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': LinearSVC(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
    def train_and_evaluate(self, model_name, X_train_vec, y_train, X_test_vec, y_test, 
                           vectorizer=None, save_model=False, filepath="best_model.pkl"):
        """Train 1 thuật toán do người dùng truyền vào, có lựa chọn save model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ. Các model khả dụng: {list(self.available_models.keys())}")
            
        model = self.available_models[model_name]
        
        model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        if save_model:
            if vectorizer is None:
                raise ValueError("Missing 'vectorizer' argument to save pipeline (model + vectorizer).")
            
            pipeline_data = {
                'model': model,
                'vectorizer': vectorizer
            }
            joblib.dump(pipeline_data, filepath)
            print(f"Saved '{model_name}' at '{filepath}'")
            
        return model, y_pred, accuracy, report

    @staticmethod
    def predict_single_text(text, filepath="best_model.pkl"):
        pipeline_data = joblib.load(filepath)
        model = pipeline_data['model']
        vectorizer = pipeline_data['vectorizer']
        
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        
        probabilities = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vec)[0]
            classes = model.classes_
            probabilities = dict(zip(classes, probs))
            
        return prediction, probabilities


class ModelComparator:
    def __init__(self, models_to_compare):
        """
        models_to_compare: dictionary contain name and instance.
        Example: {'LogReg': LogisticRegression(), 'SVM': LinearSVC()}
        """
        self.models_to_compare = models_to_compare
        
    def compare(self, X_train_vec, y_train, X_test_vec, y_test):
        """Train and rank all models provided"""
        results = []
        
        for name, model in self.models_to_compare.items():
            print(f"Processing algorithm: {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            results.append({'Model': name, 'Accuracy': acc})
            
        df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        return df_results
    


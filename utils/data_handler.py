import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataHandler:
    def __init__(self, filepath, text_col='text', label_col='class'):
        self.filepath = filepath
        self.text_col = text_col
        self.label_col = label_col
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        
    def load_and_split(self, test_size=0.2, random_state=42):
        df = pd.read_csv(self.filepath).dropna(subset=[self.text_col, self.label_col])
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        X = df[self.text_col]
        y = df[self.label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
        
    def vectorize_data(self, X_train, X_test):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec
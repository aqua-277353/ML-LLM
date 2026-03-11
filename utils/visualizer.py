import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes=['Neg', 'Pos'], title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def plot_model_comparison(df_results):
        """Draw barchart comparing the Accuracy of different models"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Accuracy', y='Model', data=df_results, palette='viridis')
        plt.title('Comparison of Accuracy for Machine Learning Models')
        plt.xlim(0, 1.0) # Accuracy ranges from 0 to 1
        
        # Display the values directly on the bars
        for index, value in enumerate(df_results['Accuracy']):
            plt.text(value + 0.01, index, f"{value:.4f}", va='center')
            
        plt.show()
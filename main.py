from src.train import train_model
from src.dataset import save_iris_to_csv
from src.evaluate import evaluate_model

if __name__ == "__main__":
    save_iris_to_csv()  # Save Iris dataset to CSV
    train_model()       # Run training pipeline
    evaluate_model()    # <- shows classification report + confusion matrix
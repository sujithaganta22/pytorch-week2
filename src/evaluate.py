import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import IrisDataset
from src.model import MLP

def evaluate_model():
    dataset = IrisDataset("data/iris.csv")
    dataloader = DataLoader(dataset, batch_size=16)

    model = MLP()
    model.load_state_dict(torch.load("model.pth", weights_only=True))  

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y.numpy())

    print("Classification Report:\n", classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

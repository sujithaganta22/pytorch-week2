import pandas as pd
import torch
from sklearn.datasets import load_iris
from torch.utils.data import Dataset

def save_iris_to_csv():
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target], axis=1)
    df.columns = list(data.feature_names) + ["target"]
    df.to_csv("data/iris.csv", index=False)

class IrisDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.drop("target", axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
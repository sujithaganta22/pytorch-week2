import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.dataset import IrisDataset
from src.model import MLP

def train_model():
    dataset = IrisDataset("data/iris.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for X, y in dataloader:
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        torch.save(model.state_dict(), "model.pth")

# PyTorch-Powered Machine Learning Project – Week 2

## 🔍 Overview

This project demonstrates a complete machine learning pipeline using **PyTorch**, starting from data preprocessing to model evaluation and hyperparameter tuning. The goal was to classify Iris flower species based on sepal and petal measurements using a Multi-Layer Perceptron (MLP).

## 📁 Project Structure

pytorch-week2/
│
├── data/
│ └── iris.csv # Dataset used
├── src/
│ ├── dataset.py # Custom Dataset and DataLoader
│ ├── model.py # PyTorch neural network model
│ ├── train.py # Training loop logic
│ └── evaluate.py # Evaluation and metrics
│
├── main.py # Orchestrates loading, training, and evaluation
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignores virtual environment and cache files

## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sujithaganta22/pytorch-week2.git
cd pytorch-week2
Create and activate a virtual environment:


python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate      # On Mac/Linux
Install dependencies:


pip install -r requirements.txt
Run the pipeline:


python main.py
📊 Model Details
Framework: PyTorch

Task: Multiclass classification

Dataset: Iris dataset

Model: Fully Connected Neural Network (3-layer MLP)

Loss Function: CrossEntropyLoss

Optimizer: Adam

Metrics: Accuracy, Precision, Recall, F1-score

🔁 Hyperparameter Tuning
We experimented with the following:

Learning Rate: 0.001, 0.005, 0.01

Batch Size: 16, 32

Hidden Layers: 1, 2, 3

The best model achieved 91% accuracy on the test set.

🤖 GitHub Copilot Usage
GitHub Copilot was used to:

Generate PyTorch class structures (Dataset, nn.Module)

Suggest training loop templates

Write helper functions like accuracy_score and evaluation logic

All Copilot suggestions were verified, edited, and documented to ensure correctness.

📚 References
UCI ML Repository: Iris Data Set

PyTorch Documentation: https://pytorch.org/docs/stable/index.html

scikit-learn Documentation: https://scikit-learn.org/stable/

GitHub Copilot: https://github.com/features/copilot

📌 Author: Sujitha Ganta
📅 Submitted for: Machine Learning with PyTorch - Week 2 Assignment
📬 For feedback or questions, open an issue or email the author.

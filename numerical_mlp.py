import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(y_preds, y_trues):
    total = [y_pred == y_true for (y_pred, y_true) in zip(y_preds, y_trues)]
    acc = sum(total) / len(y_trues)
    return acc

class TabularDataset(Dataset):
    def __init__(self, X, Y=None):
        self.n = len(X)
        self.y = None
        if Y is not None:
            self.y = Y.values
        normalized_X_num = (X-X.min())/(X.max() - X.min())
        self.x = normalized_X_num.astype(np.float32).values

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.y is not None:
            return [self.x[idx], self.y[idx]]
        return self.x[idx]

    
class TabularMLP(nn.Module):
    def __init__(self, num_input_size, hidden_size, target_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout1 = torch.nn.Dropout(.1)
        self.dropout2 = torch.nn.Dropout(.1)
        self.output = torch.nn.Linear(hidden_size, target_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        y_ = self.output(x)
        return y_

class MLP():
    def __init__(self, num_input_size, hidden_size, target_size):
        self.model = TabularMLP(num_input_size, hidden_size, target_size)
        self.error = []
        self.accuracy = []

    def fit(self, X, Y, **kargs):
        self.error = []
        self.accuracy = []
        
        epochs = kargs.get('epochs', 100)
        lr = kargs.get('lr', 0.01)
        batch_size = kargs.get('batch_size', 64)

        model = self.model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        training_dataset = TabularDataset(X=X, Y=Y)
        dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            print("\rÉpoca {}/{}".format(epoch+1, epochs), end='')
            total_loss = 0
            results = []
            target = []
            for X, Y in dataloader:    

                X = X.to(DEVICE)
                Y = Y.to(DEVICE)
                
                # Forward pass
                Y_ = model(X)
                loss = criterion(Y_, Y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient descent
                optimizer.step()

                total_loss += loss.item()*X.size(0)
                
                results.extend([np.argmax(y) for y in Y_.tolist()])
                target.extend(Y.tolist())
                
            self.error.append(total_loss/len(training_dataset))
            self.accuracy.append(calculate_accuracy(results, target))
            
    def predict(self, X):
        model = self.model.to(DEVICE)
        model.eval()
        
        dataset = TabularDataset(X=X)
        test_dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        total_loss = 0
        results = []
        for X in test_dataloader: 
            X = X.to(DEVICE)
            Y_ = model(X)
            results.extend([np.argmax(y) for y in Y_.tolist()])
        return results
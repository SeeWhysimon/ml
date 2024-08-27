import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import optimizer

def train(data_loader: DataLoader, 
          model: torch.nn.Module, 
          optimizer: optimizer, 
          device: torch.device):
    model.train()
    
    for data in data_loader:
        reviews = data["review"]
        targets = data["target"]

        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        predictions = model(reviews)

        loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))
        loss.backward()
        optimizer.step()

def evaluate(data_loader: DataLoader,
             model: nn.Module,
             device: torch.device):
    final_predictions = []
    final_targets = []

    model.eval()

    with torch.no_grad():
        for data in data_loader:
            reviews = data["review"]
            targets = data["target"]
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            predictions = model(reviews)

            predictions = predictions.cpu().numpy().tolist()
            targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
            
    return final_predictions, final_targets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from alexnet_dataset import CatDogDataset
from alexnet import AlexNet
from alexnet_engine import train_one_epoch, validate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = CatDogDataset(csv_file='../../data/cat_dog/train.csv')

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = AlexNet(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')

        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

    torch.save(model.state_dict(), '../../results/Classification/alexnet_cat_dog.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()

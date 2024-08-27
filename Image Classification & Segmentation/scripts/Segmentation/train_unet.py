import torch
import torch.nn as nn
import torch.optim as optim
from unet import UNet
from voc_dataset import get_train_loader

# UNet模型定义
def double_conv(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def train():
    num_epochs = 20
    model = UNet()
    train_loader = get_train_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images
            masks = masks.squeeze(1).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), '../../../results/Image Classification & Segmentation/Segmentation/unet_model.pth')

if __name__ == "__main__":
    train()
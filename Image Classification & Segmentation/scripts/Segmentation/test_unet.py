import torch
from PIL import Image
import matplotlib.pyplot as plt
from train_unet import UNet
from torchvision import transforms

def load_model():
    model = UNet()
    model.load_state_dict(torch.load('../../results/Segmentation/unet_model.pth'))
    model.eval()
    model
    return model

def test():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_model()

    test_image_path = '../../data/VOC2012_train_val/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'
    test_image = Image.open(test_image_path)
    test_image = transform(test_image).unsqueeze(0)

    with torch.no_grad():
        output = model(test_image)
        output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    plt.imshow(output)
    plt.axis('off')
    plt.savefig('../../results/Segmentation/unet_output.png')

if __name__ == "__main__":
    test()

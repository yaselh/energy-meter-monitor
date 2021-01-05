import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
num_epochs = 20
num_classes = 20
batch_size = 32
learning_rate = 1e-3
input_size = 68

class LetNet5(nn.Module):
    def __init__(self, num_clases=10):
        super(LetNet5, self).__init__()

        self.c00 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c3 = nn.Sequential(
            nn.Conv2d(32, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )
 
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
 
        self.fc2 = nn.Sequential(
            nn.Linear(84, num_classes),
            nn.LogSoftmax()
        )

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(input_size, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
            ])
 
    def forward(self, x):
        out = self.c00(x)
        out = self.c1(out)
        out = self.c2(out)
        out = self.c3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def prepare_dataset(self):
        # dataset
        train_dataset = torchvision.datasets.ImageFolder("data/scut/preprocessed/easy_samples/train/", 
                                            transform=self.transforms)
        
        test_dataset = torchvision.datasets.ImageFolder("data/scut/preprocessed/easy_samples/test/", 
                                            transform=self.transforms)        
        # Data loader
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size)
        
    def save(self):
        torch.save(self.state_dict(), 'LetNet-5_v1.ckpt')

    def load(self):
        path_ckpt = "./models/cnn/weights/LetNet-5_v1.ckpt"
        checkpoint = torch.load(Path(path_ckpt), map_location=device)
        self.load_state_dict(checkpoint)

    def fit(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)
        
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
        
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    def evaluate(self): 
        # Test the model
        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    
    def predict(self, image):
        self.eval()
        with torch.no_grad():
            image = Image.fromarray(image)
            input = self.transforms(image).to(device)
            input = input.view(1, 1, input_size, input_size)
            output = self(input)
            logprob, predicted = torch.max(output.data, 1)
            predicted = predicted.cpu().numpy()[0]
            confidence= torch.exp(logprob).cpu().numpy()[0]
            return predicted, confidence

if __name__ == "__main__":
    mode = "train"
    model = LetNet5(num_classes).to(device)
    if mode == "train":
        model.prepare_dataset()
        model.fit()
        model.evaluate()
        model.save()
    else:
        model.load()
        #model.predict(img)

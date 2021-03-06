import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

batch_size = 16

transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.Grayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5],
                                                      [0.5])
                                 ])

train_data = datasets.ImageFolder(root='flower_photos/train/',
                                  transform=transforms)

test_data = datasets.ImageFolder(root='flower_photos/test/',
                                 transform=transforms)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self,):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 50176)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


model = AutoEncoder()
if torch.cuda.is_available():
    model = model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 130

for epoch in range(n_epochs):
    train_loss = 0.0
    for data, target in train_loader:
        if torch.cuda.is_available():
            data = data.cuda()
        data = data.view(data.size(0), -1)
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    print("Epoch {} loss {}".format(epoch+1, train_loss/len(train_loader)))


images, labels = next(iter(test_loader))
x_flat = images.view(images.size(0), -1)
out_recon = model.forward(x_flat.cuda())
out_recon = out_recon.view(batch_size, 224, 224, 1)
out_recon = out_recon.cpu().detach().numpy()


fig = plt.figure(figsize=(10, 5))
columns = 4
rows = 2

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.squeeze(out_recon[i]), cmap='gray')

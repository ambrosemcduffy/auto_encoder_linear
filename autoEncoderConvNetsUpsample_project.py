import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

batch_size = 16

transforms = transforms.ToTensor()

train_data = datasets.MNIST(root='data',
                            train=True,
                            download=True,
                            transform=transforms)

test_data = datasets.MNIST(root='data',
                           train=False,
                           download=True,
                           transform=transforms)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

data, target = next(iter(train_loader))

plt.imshow(np.squeeze(data[0]), cmap='gray')


class AutoEncoder(nn.Module):
    def __init__(self,):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1 , 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.sigmoid(self.conv4(x))
        return x


model = AutoEncoder()
if torch.cuda.is_available():
    model = model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
noise_factor = 50

for epoch in range(n_epochs):
    train_loss = 0.0
    for data, target in train_loader:
        noisy_imgs = data + noise_factor * torch.randn(*data.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        if torch.cuda.is_available():
            data = data.cuda()
        #data = data.view(data.size(0), -1)
        output = model(noisy_imgs.cuda())
        optimizer.zero_grad()
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    print("Epoch {} loss {}".format(epoch+1, train_loss/len(train_loader)))


images, labels = next(iter(test_loader))
#x_flat = images.view(images.size(0), -1)
out_recon = model.forward(images.cuda())
out_recon = out_recon.view(batch_size, 28, 28, 1)
out_recon = out_recon.cpu().detach().numpy()


fig = plt.figure(figsize=(10, 5))
columns = 4
rows = 2

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.squeeze(out_recon[i]), cmap='gray')
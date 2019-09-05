import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torchvision.utils import save_image

# Parameter to change
num_datasets = 10
dataset_size = 250
epochs = 500

f = h5py.File('mnist_dataset.hdf5', 'r')
x_test = torch.Tensor(np.array(f["test/x"])).cuda()
y_test = torch.Tensor(np.array(f["test/y"])).long().cuda()

class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VAE(nn.Module):
    def __init__(self, x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


def train_model():
    global x_test, y_test
    test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=True)

    models = []
    for dataset_i in range(num_datasets):
        print(dataset_i)
        x_train = torch.Tensor(np.array(f["train_%d/x" % dataset_i])).cuda()
        y_train = torch.Tensor(np.array(f["train_%d/y" % dataset_i])).cuda()
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
        model = Net().cuda()
        model.train()
        opt = torch.optim.Adam(model.parameters())
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                opt.zero_grad()
                pred = model(xb)
                loss = torch.nn.functional.nll_loss(pred, yb.long())
                loss.backward()
                opt.step()
            # if epoch % 10 == 0:
            #     print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                pred = model(xb).argmax(dim=1)
                correct += pred.eq(yb.view_as(pred)).sum().item()
        print('Accuracy: {:.6f}'.format(float(correct) / len(test_ds)))

        models.append(model)

    torch.save({'model_{}'.format(i): models[i].state_dict() for i in range(num_datasets)}, 'mnist_models.pth')

def train_vae():
    global x_test, y_test
    test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=True)

    def loss_function(recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    models = []
    for dataset_i in range(num_datasets):
        print(dataset_i)
        x_train = torch.Tensor(np.array(f["train_%d/x" % dataset_i])).cuda()
        y_train = torch.Tensor(np.array(f["train_%d/y" % dataset_i])).cuda()
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
        model = VAE().cuda()
        model.train()
        opt = torch.optim.Adam(model.parameters())
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_dl:
                opt.zero_grad()
                xb_recon, mu, log_var = model(xb)
                loss = loss_function(xb_recon, xb, mu, log_var)
                train_loss += loss.item()
                loss.backward()
                opt.step()
            if epoch % 10 == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / len(train_ds)))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb_recon, mu, log_var = model(xb)
                test_loss += loss_function(xb_recon, xb, mu, log_var)
        test_loss /= len(test_ds)
        print('Test loss: {:.6f}'.format(test_loss))

        # with torch.no_grad():
        #     z = torch.randn(64, 2).cuda()
        #     sample = model.decoder(z).cuda()
        #
        #     save_image(sample.view(64, 1, 28, 28), 'sample_' + str(dataset_i) + '.png')

        models.append(model)

    torch.save({'vae_{}'.format(i): models[i].state_dict() for i in range(num_datasets)}, 'mnist_vaes.pth')

# train_model()
train_vae()
f.close()
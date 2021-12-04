from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        
class Net_FC(nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
                
def train_momentum_spiderboost(model, device, train_loader_xi, optimizer, epoch, b_sz):
    model.train()
    
    q = 64
    B = 64
    eta = 0.05
    v2 = {}
    v = {}
    v3 = {}
    iter = 0
    lmbda = 0.00001
    for batch_idx, (data, target) in enumerate(train_loader_xi):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
               
        if iter % q == 0:
            for name, p in model.named_parameters():
                v[name] = p.grad.data
                v2[name] = p.data       
        else:
            #v1 is storing gradients, v2 i storing old_x and v3 is storing current_x
            v1 = {}
            for p in model.parameters():
                p.grad.data.add_(p.grad.data)
            
            for name, p in model.named_parameters():
                v1[name] = p.grad.data
                v3[name] = p.data
                p.data.copy_(v2[name])
            
            data = data.to(device)  
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            
            for p in model.parameters():
                p.grad.data.add_(p.grad.data)
                
            for name, p in model.named_parameters():
                    v1[name].add_(-p.grad.data)
                    v1[name].add_(v[name])
            v = v1
            
            for name, p in model.named_parameters():
                v[name].mul_(1/B)
            
            for name, p in model.named_parameters():
                p.data.copy_(v3[name])
                
            data = data.to(device)  
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
        
        for name, p in model.named_parameters():
            p.data.add_(-eta*v[name])
            p.data = torch.sign(p.data) * torch.maximum(torch.abs(p.data)-lmbda,\
                    torch.zeros(p.data.size()))
        v2 = v3  
            
        iter +=1 
        
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_xi.dataset),
                100. * batch_idx / len(train_loader_xi), loss.item()))

def test(model, device, test_loader,y,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    y[epoch] = 100. * correct / len(test_loader.dataset)
    
def modelSparsity(model):
    nonzeros = 0
    parameterCount = 0
    
    for p in model.parameters():
        for data in torch.flatten(p.data):
            if(data!=0):nonzeros+=1
            parameterCount += 1
    
    return 1 - (nonzeros/parameterCount)

def stationarityViolation(model):
    lmbda = 0.00001
    gradient = torch.tensor(())
    for p in model.parameters():
        data = torch.flatten(p.data)
        grad = torch.flatten(p.grad.data)
        gradient = torch.cat((gradient,(torch.sign(data-grad)*torch.maximum\
                              (torch.zeros(data.size()), torch.abs(data-grad)-lmbda))))
    return float(torch.linalg.norm(grad))    

def main():
    # Training settings
    use_cuda = False # if your machine has CUDA-compatible GPU, you can change this to True

    torch.manual_seed(20200930)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader_xi = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    
    
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True, **kwargs)
    
    #model = Net_FC().to(device)
    
    model = LeNet5().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    iter = 0
    
    b_sz = 64
    
    totalEpochs = 15
    x = np.zeros(totalEpochs)
    y = np.zeros(totalEpochs)
    z = np.zeros(totalEpochs)
    w = np.zeros(totalEpochs)
    for epoch in range(totalEpochs):
        #train(model, device, train_loader, optimizer, epoch, iter)
        train_momentum_spiderboost(model, device, 
                       train_loader_xi, optimizer, epoch, b_sz)
        test(model, device, test_loader,y,epoch)
        iter += 60000//64
        
        x[epoch] = epoch
        z[epoch] = modelSparsity(model)
        w[epoch] = stationarityViolation(model)

    fig, axs = plt.subplots(2,1)
    fig.suptitle("Accuracy, sparsity and violation")
    axs[0].plot(x, y, '+') 
    plt.ylim(75,90)
    axs[1].plot(x, z, '*')  
    #axs[2].plot(x, w, '+') 
    
        
if __name__ == '__main__':
    main()

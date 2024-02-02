import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

train_set = torchvision.datasets.MNIST("./data", train=True, download=True,
                                       transform = torchvision.transforms.ToTensor())
test_set = torchvision.datasets.MNIST("./data", train=False, download=True,
                                       transform = torchvision.transforms.ToTensor())

train_dataset = DataLoader(train_set, batch_size=256)
test_dataset = DataLoader(test_set, batch_size=256)


EPOCH = 30
LR = 0.01
batch_size = 256


# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, input):
        input = input.reshape((-1, 28*28))
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        output = self.fc3(input)
        return output


# Initialize the MLP model
model = MLP()


# Specify loss function
loss_function = nn.CrossEntropyLoss()

# Specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# Train the model

for epoch in range(EPOCH):
    train_loss = 0.0

    for data, target in train_dataset:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)

    train_loss = train_loss / len(train_dataset.dataset)

    print('Epoch: {} \tTraning Loss: {:.6f}'.format(epoch+1, train_loss))


#  Test the model

test_loss = 0.0
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))

for data, target in test_dataset:
    output = model(data)
    loss = loss_function(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    for i in range(data.size(0)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss / len(test_dataset.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of {i}: %2d%% (%2d/%2d)' % (
            100 * class_correct[i] / class_total[i],
            class_correct[i], class_total[i]
        ))

    else:
        print('There is no number {i}\n')

print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100 * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)
))














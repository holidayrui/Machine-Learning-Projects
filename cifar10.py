import torch.cuda
import time
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# Prepare dataset
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("The length of training dataset is: {}".format(train_data_size))
print("The length of testing dataset is: {}".format(train_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class Mode(nn.Module):
    def __init__(self):
        super(Mode, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    mode = Mode()
    input = torch.ones((64, 3, 32, 32))
    output = mode(input)
    print(output.shape)


# Construct the neural network
mode = Mode()
if torch.cuda.is_available():
    mode = mode.cuda()

# loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# optim
learning_rate = 0.01
optimizer = torch.optim.SGD(mode.parameters(), lr=learning_rate)

# set network parameters for training
total_train_step = 0
total_test_step = 0
epoch = 30

# Use tensorboard to see the graph
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------{} time of training------- ".format(i + 1))

    # The part of training
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = mode(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("Times of training: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # The part of testing
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = mode(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("The total loss for test dataset is: {}".format(total_test_loss))
    print("The accurarcy of the test dataset is: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(mode, "mode_{}.pth".format(i+1))

writer.close()

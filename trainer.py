import torch
import torch.optim as optim
import torch.nn as nn

import model
import Dataloader

# To check if cuda is available, and use it.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

val = Dataloader.loadData()

# Initialize the LeNet model for training
model = model.LeNet()
model.to(device)

# Defining the loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to train the data for a given number of epochs


def train(n):

    for epoch in range(n):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(val.trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print the loss details for epochs
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(
                    f'epoch no :{epoch + 1} batch no :{i+1} loss : {running_loss/2000}')
                running_loss = 0.0

    print('\n Finished Training')

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    train(1)

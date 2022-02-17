import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn


# read image
def read_img(path_fork, path_spoon, flag):
    # True for training and validation, False for testing.
    if flag is True:
        img_dir_fork = sorted(os.listdir(path_fork))
        img_dir_spoon = sorted(os.listdir(path_spoon))
        x = np.zeros((len(img_dir_fork)+len(img_dir_spoon), 227, 227, 3), dtype=np.uint8)
        y = np.zeros((len(img_dir_fork)+len(img_dir_spoon)), dtype=np.uint8)
        print(str(len(img_dir_fork)+len(img_dir_spoon)))
        for i, file in enumerate(img_dir_fork):
            x[i, :, :] = cv2.resize(cv2.imread(os.path.join(path_fork, file)), (227, 227))
            y[i] = 0
        for i, file in enumerate(img_dir_spoon):
            x[i+len(img_dir_fork), :, :] = cv2.resize(cv2.imread(os.path.join(path_spoon, file)), (227, 227))
            y[i+len(img_dir_fork)] = 1
        return x, y
    elif flag is False:
        img_dir = sorted(os.listdir(path_fork))
        print("test:" + str(len(img_dir)))
        x = np.zeros((len(img_dir), 227, 227, 3), dtype=np.uint8)
        for i, file in enumerate(img_dir):
            x[i, :, :] = cv2.resize(cv2.imread(os.path.join(path_fork, file)), (227, 227))
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input 3x227x227
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
                                 nn.BatchNorm2d(96),  # 96x55x55
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # 96x27x27

                                 nn.Conv2d(96, 256, 5, 1, 2),  # 256x27x27
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, 2, 0),  # 256x13x13

                                 nn.Conv2d(256, 384, 3, 1, 1),  # 384x13x13
                                 nn.BatchNorm2d(384),
                                 nn.ReLU(),

                                 nn.Conv2d(384, 384, 3, 1, 1),  # 384x13x13
                                 nn.BatchNorm2d(384),
                                 nn.ReLU(),

                                 nn.Conv2d(384, 256, 3, 1, 1),  # 256x13x13
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3, 2, 0),  # 256x6x6
                                 )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)

        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return self.dropout(out)


# data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataSet(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_get = self.x[index]
        if self.transform is not None:
            x_get = self.transform(x_get)
        if self.y is not None:
            y_get = self.y[index]
            return x_get, y_get
        else:
            return x_get


path_training_spoon = "D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\train_spoon"
path_training_fork = "D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\train_fork"
path_validation_spoon = "D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\val_spoon"
path_validation_fork = "D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\val_fork"
train_x, train_y = read_img(path_training_fork, path_training_spoon, True)
val_x, val_y = read_img(path_validation_fork, path_validation_spoon, True)
batch_size = 32
train_set = ImgDataSet(train_x, train_y, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = ImgDataSet(val_x, val_y, test_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = Net().cuda()
loss = nn.CrossEntropyLoss()  # classification problem
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epoch = 30
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    for i, data in enumerate(train_loader):
        model.train()  # set the module in training mode
        optimizer.zero_grad()
        train_prediction = model(data[0].cuda())
        batch_loss = loss(train_prediction, data[1].cuda())
        batch_loss.backward()
        optimizer.step()
        train_acc += np.sum(np.argmax(train_prediction.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()  # set the module in evaluation mode
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_prediction = model(data[0].cuda())
            batch_loss = loss(val_prediction, data[1].cuda())
            val_acc += np.sum(np.argmax(val_prediction.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        print('[%03d/%03d]  Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' %
              (epoch + 1, num_epoch, train_acc / train_set.__len__(), train_loss / train_set.__len__(),
               val_acc / val_set.__len__(), val_loss / val_set.__len__()))

# train with all images(training set + validation set)
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataSet(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    for i, data in enumerate(train_val_loader):
        model.train()
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    print('[%03d/%03d]  Train Acc: %3.6f Loss: %3.6f' % (epoch + 1, num_epoch, train_acc / train_val_set.__len__(),
                                                         train_loss / train_val_set.__len__()))

torch.save(model.state_dict(), 'D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\CNN_fs.pt')


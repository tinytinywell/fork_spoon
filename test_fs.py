import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


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


# def read_img(path):
#     img_dir = os.listdir(path)
#     x = np.zeros((len(img_dir), 227, 227, 3), dtype=np.uint8)
#     for i, file in enumerate(img_dir):
#         image = cv2.imread(os.path.join(path, file))
#         x[i, :, :] = cv2.resize(image, (227, 227))
#     return x


# path_testing = "D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\te"
# test_x = read_img(path_testing)
# batch_size = 32
# test_set = ImgDataSet(test_x, transform=test_transform)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# prediction = []

# load the model
model = Net().cuda()
model_state_dict = torch.load('D:\\PycharmProjects\\Machine Learning\\PythonMachineLearning\\spoon-vs-fork\\CNN_fs.pt')
model.load_state_dict(model_state_dict)
model.eval()

# with torch.no_grad():
#     p_index = []
#     for i, data in enumerate(test_loader):
#         test_pred = model(data.cuda())
#         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         for y in test_label:
#             prediction.append(y)
# print(prediction)
# label = np.zeros(len(test_x), dtype=np.uint8)
# print(np.sum(prediction == label)/len(test_x))

# cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam = cv2.VideoCapture("D://PycharmProjects//Machine Learning//PythonMachineLearning//spoon-vs-fork//video//test5.mp4")
# fps = cam.get(cv2.CAP_PROP_FPS)
fps = 30
size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
four_cc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
outVideo =cv2.VideoWriter(
    "D://PycharmProjects//Machine Learning//PythonMachineLearning//spoon-vs-fork//video//test_r.mp4",
                          four_cc, fps, size)
while True:
    ret, img = cam.read()
    frame = np.zeros((1, 227, 227, 3), dtype=np.uint8)
    frame[0, :, :] = cv2.resize(img, (227, 227))
    frame_set = ImgDataSet(frame, transform=test_transform)
    frame_loader = DataLoader(frame_set, batch_size=1, shuffle=False)
    for r in frame_loader:
        prediction = model(r.cuda())
        m = nn.Sigmoid()
        test_label = np.argmax(prediction.cpu().data.numpy(), axis=1)
        pro = (m(prediction).cpu().data.numpy())[0][test_label]
        if test_label == 0:
            text = "Fork: "
        elif test_label == 1:
            text = "Spoon: "
        cv2.putText(img, text + str(pro), (100, 109), cv2.QT_FONT_NORMAL, 1.2, (255, 255, 255), 2)
    cv2.imshow("press q to exit", img)
    outVideo.write(img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
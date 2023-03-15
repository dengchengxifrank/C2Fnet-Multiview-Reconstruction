import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from common import normalize_imagenet
from common import export_pointcloud
import tempfile
import subprocess
import os
import trimesh
import numpy

from pointnet2 import*

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'



net = models.resnet50(weights=None)

net.load_state_dict(torch.load('resnet50-0676ba61.pth')) 


class mlps(nn.Module):
    def __init__(self):
        super().__init__()
        #后面需要与general feature 拼接

        self.mlp1 = nn.Linear(1024,512)

        self.mlp2 = nn.Linear(1024,256)

        self.mlp3 = nn.Linear(512,256)

        self.mlp4 = nn.Linear(320,256)

        self.mlp5 = nn.Linear(320,320)
        self.relu = nn.ReLU()
        self.Dropout1 = nn.Dropout(0.5)
        self.Dropout2 = nn.Dropout(0.5)
        self.Dropout3 = nn.Dropout(0.5)
        self.Dropout4 = nn.Dropout(0.5)


    def forward(self,x):

        x = self.relu(self.mlp1(x))

        x = self.Dropout(x)

        return x



# class decodermlp(nn.Module):
#     def __init__(self):
#         super().__init__()


class coarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout()

        self.fc_0 = nn.Linear(1000, 512)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_3 = nn.Linear(512, 1024 * 3)

    def forward(self, x):
        batchsize = x.size(0)

        x = self.net(x)

        x = self.fc_0(x)

        x = self.fc_1(self.Dropout(self.relu(x)))

        x = self.fc_2(self.Dropout(self.relu(x)))

        x = self.fc_3(self.relu(x))

        x = x.view(batchsize, 1024, 3)

        return x



#现在先focus到三张图片重建上面

class imagecov(nn.Module):
    def __init__(self):
        super(imagecov, self).__init__()
        #第一层
        self.conv1 = nn.Conv2d(in_channels=9,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.BatchNormal1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)


        #第二层
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.BatchNormal2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

        #第三层
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.BatchNormal3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

        #第四层
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.BatchNormal4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

        #第五层
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.BatchNormal5 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

        #第六层
        self.conv6 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.BatchNormal6 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

        #第七层
        self.conv7 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.BatchNormal7 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        #第八层
        self.conv8 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.BatchNormal8 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

        #embedding 1
        self.linear1 = nn.Linear(3136,512)

        self.linear2 = nn.Linear(288,512)

        self.linear3 = nn.Linear(288,256)

        self.linear4 = nn.Linear(144,64)

        self.linear5 = nn.Linear(144,64)


    def forward(self,x):
        x = self.conv1(x)
        x = self.BatchNormal1(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.BatchNormal2(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv3(x)
        x = self.BatchNormal3(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv4(x)
        x = self.BatchNormal4(x)
        x = self.relu(x)
        x = self.pooling(x)


        x = self.conv5(x)
        x = self.BatchNormal5(x)
        x = self.relu(x)
        x = self.pooling(x)

        x1 = self.linear1(x.view(-1))

        x = self.conv6(x)
        x = self.BatchNormal6(x)
        x = self.relu(x)
        x = self.pooling(x)

        x2 = self.linear2(x.view(-1))

        x = self.conv7(x)
        x = self.BatchNormal7(x)
        x = self.relu(x)

        x3 = self.linear3(x.view(-1))

        x = self.conv8(x)
        x = self.BatchNormal8(x)
        x = self.relu(x)

        x4 = self.linear4(x.view(-1))

        x5 = self.linear5(x.view(-1))


        return x1, x2, x3, x4, x5



class coarse_to_fine(nn.Module):

    def __init__(self):
        super().__init__()
        self.coarse = coarse()
        self.imagecov = imagecov()
        self.get_model = get_model(512)

        self.mlp1 = nn.Linear(1024,512)
        self.mlp2 = nn.Linear(1024,256)
        self.mlp3 = nn.Linear(512,256)
        self.mlp4 = nn.Linear(320,256)
        self.mlp5 = nn.Linear(320,320)

        self.relu = nn.ReLU()

        self.Dropout1 = nn.Dropout(0.5)
        self.Dropout2 = nn.Dropout(0.5)
        self.Dropout3 = nn.Dropout(0.5)
        self.Dropout4 = nn.Dropout(0.5)
        self.Dropout5 = nn.Dropout(0.5)

        #mlp as the finnal decoder

        self.finnal_decoder = nn.Sequential(
            nn.Linear(960,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024*2,1024*3)
        )

    def forward(self,x):
        x = x.squeeze(dim=0)
        #现在拿到的是coarse的点云
        image1 = x[0]
        image2 = x[1]
        image3 = x[2]
        input = torch.cat((image1, image2, image3), dim=0)

        input = input.unsqueeze(0)

        x1, x2, x3, x4, x5 = self.imagecov(input)
        x1 = x1.repeat(3,1)
        x2 = x2.repeat(3, 1)
        x3 = x3.repeat(3, 1)
        x4 = x4.repeat(3, 1)
        x5 = x5.repeat(3, 1)

        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        pointcloud = self.coarse(x)

        pointcloud = pointcloud.transpose(2, 1)

        point_feature = self.get_model(pointcloud)
         
        
        point1_feature , point2_feature , point3_feature= point_feature[0],point_feature[1],point_feature[2]

        #print(point1_feature.shape,point2_feature.shape,point3_feature.shape)

        feature_concat1 = torch.cat((x1,point_feature),dim=1)
        output1 = self.Dropout1(self.relu(self.mlp1(feature_concat1)))
        output1 = torch.cat((x2,output1),dim=1)

        output2 = self.Dropout2(self.relu(self.mlp2(output1)))
        output2 = torch.cat((x3,output2),dim=1)

        output3 = self.Dropout3(self.relu(self.mlp3(output2)))
        output3 = torch.cat((x4,output3),dim=1)

        output4 = self.Dropout4(self.relu(self.mlp4(output3)))
        output4 = torch.cat((x5,output4),dim=1)

        output5 = self.Dropout5(self.relu(self.mlp5(output4)))

        #这里相当于batchsize为3
        output5 = output5.view(-1)

        result_point = self.finnal_decoder(output5).view(1024,3)
        
        #print('res point is',result_point.shape)
        
        
        #result_feature = self.get_model(result_point)
        
        
        return result_point,point1_feature,point2_feature,point3_feature






    # model = coarse()
    #
    # res = model(image)
    #
    # res = res.detach().numpy()
    #
    # numpy.save("../result/1.npy",res)


from __future__ import print_function
import argparse
import os
import random
import torch
from os.path import join, exists, isdir, dirname, abspath, basename
import json
from dataset_test import GetShapenetDataset
import torch.backends.cudnn as cudnn
#from model import generator
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model1 import*
from coarse_to_fine_cl import *

from test_iou import *

from chamfer_distance import ChamferDistance

chamfer_dist = ChamferDistance()

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--cats', default=['03691459'], type=str,
                    help='Category to train on : ["airplane":02691156, "bench":02828884, "cabinet":02933112, '
                         '"car":02958343, "chair":03001627, "lamp":03636649, '
                         '"monitor":03211117, "rifle":04090263, "sofa":04256520, '
                         '"speaker":03691459, "table":04379243, "telephone":04401088, '
                         '"vessel":04530566]')
parser.add_argument('--num_points', type=int, default=1024, help='umber of pointcloud, [1024, 2048]')


opt = parser.parse_args()
#parser.add_argument('--model', type=str, default='./model/best.pth',  help='generator model path')

# with open(join('data/splits/', 'train_models.json'), 'r') as f:
#     train_models_dict = json.load(f)


# with open(join('data/splits/', 'val_models.json'), 'r') as f:
#     val_models_dict = json.load(f)


with open(join('./datasplit/' + opt.cats[0] + '_train.json'), 'r') as f:
    train_models_dict = json.load(f)
    
with open(join('./datasplit/' + opt.cats[0]  + '_test.json'), 'r') as f:
    #val_models_dict = json.load(f)
    test_models_dict = json.load(f)
    

data_dir_imgs = './image/'

data_dir_pcl = './ShapeNet_pointclouds/'

opt = parser.parse_args()


#test_dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, val_models_dict, opt.cats, opt.num_points)
test_dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, test_models_dict, opt.cats, opt.num_points)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers),drop_last=True)


def test_cl_iou(epoch):
    
  print(epoch)
  
  model_test = coarse_to_fine().cuda()

  model_test = torch.nn.DataParallel(model_test)

  model_test.load_state_dict(torch.load('./model_03211117/model_%s_0.pth'%(epoch)))


  loss_all = []
  iou_all = []

  loss_finnal = torch.Tensor([0]).cuda()
  iou_finnal = torch.Tensor([0]).cuda()


  with torch.no_grad():
    model_test.eval()
    for i,data in enumerate(testdataloader):
        print(len(testdataloader))
        data_iter = iter(testdataloader)
 
        data = data_iter.__next__()

    
        images, points = data[0],data[1]
    
        images,points = images.cuda().squeeze(0),points.cuda().float()
        
        feature , res , coarse_point = model_test(images.to(torch.float32))        
        dist1, dist2 = chamfer_dist(points, res)

        loss = (torch.mean(dist1)) + (torch.mean(dist2))
    
        loss_all.append(loss)
        iou_value = batch_iou(res,points)
        print('the iou_value is ',iou_value)
        iou_all.append(iou_value)
   
    for i in range(len(iou_all)):
        
        iou_finnal = iou_finnal + iou_all[i]
        loss_finnal = loss_finnal + loss_all[i]
    
    print(loss_finnal/len(testdataloader))      
    print(iou_finnal/len(testdataloader))
    
    return loss_finnal,iou_finnal


    #writer.add_scalar('test loss  ', loss_cl, global_step=epoch)

    
  

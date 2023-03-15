from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from os.path import join, exists, isdir, dirname, abspath, basename
import json
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib.pyplot as plt
#from loss import batch_NN_loss
#import test_shapenet1
import os
from info_nce import *
import torch.nn.functional as F
import datetime
from tensorboardX import SummaryWriter

from test_cl import test_cl_iou

#from charmferloss import *
from dataset import GetShapenetDataset

#from coarse_fine import*

from coarse_to_fine_cl import*

from chamfer_distance import ChamferDistance

chamfer_dist = ChamferDistance()
chamfer_dist_ = ChamferDistance()

#设置显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#torch.set_printoptions(profile='full')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
# ALL: ['02691156','02828884','02933112','02958343','03001627','03636649','03211117','04090263','04256520','03691459','04379243','04401088','04530566']
parser.add_argument('--cats', default=['03211117'], type=str,
                    help='Category to train on : ["airplane":02691156, "bench":02828884, "cabinet":02933112, '
                         '"car":02958343, "chair":03001627, "lamp":03636649, '
                         '"monitor":03211117, "rifle":04090263, "sofa":04256520, '
                         '"speaker":03691459, "table":04379243, "telephone":04401088, '
                         '"vessel":04530566]')
parser.add_argument('--num_points', type=int, default=1024, help='number of epochs to train for, [1024, 2048]')
parser.add_argument('--outf', type=str, default='model',  help='output folder')
parser.add_argument('--modelG', type=str, default = '', help='generator model path')
parser.add_argument('--lr', type=float, default = '0.00001', help='learning rate')

opt = parser.parse_args()
print (opt)

#blue = lambda x:'\033[94m' + x + '\033[0m'
# seed 672
#opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 672
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)

trainfile = './datasplit/' + str(opt.cats[0]) + '_train.json'
testfile = './datasplit/' + str(opt.cats[0]) +'_test.json'
with open(trainfile, 'r') as f:
    train_models_dict = json.load(f)
    
with open(testfile, 'r') as f:
    #val_models_dict = json.load(f)
    test_models_dict = json.load(f)
    

torch.backends.cudnn.enabled = True

#data_dir_imgs = './ShapeNetRendering/'

data_dir_imgs = './image/'

data_dir_pcl = './ShapeNet_pointclouds/'

#opt.cats 是list type

dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, train_models_dict, opt.cats, opt.num_points)

print(len(dataset))

traindataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=8,drop_last=True)

test_dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, test_models_dict, opt.cats, opt.num_points)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=8,drop_last=True)

cudnn.benchmark = True

try:
    os.makedirs(opt.outf)
except OSError:
    pass

#gen = create_psgn_occu()
gen = coarse_to_fine()

# if not opt.modelG == '':
#     with open(opt.modelG, "rb") as f:
#         gen.load_state_dict(torch.load(f))

gen = torch.nn.DataParallel(gen)

#print(gen)

gen.cuda()

#gen.load_state_dict(torch.load('./model_03691459/model_1_300.pth'))

# optimizerG = optim.RMSprop(gen.parameters(), lr = opt.lr)
optimizerG = optim.Adam(gen.parameters(), lr = opt.lr,betas=(0.9, 0.999),eps=1e-08,weight_decay=0.9)

num_batch = len(dataset)/opt.batchSize

cnt = 0

term = []
loss_visual = []

this_loss = 1000000

def draw_loss(x,y):
    plt.clf()
    plt.switch_backend('agg')
    plt.plot(x,y)
    plt.savefig("./model_%s/train_loss_%s.png"%(str(opt.cats[0]),str(opt.cats[0])))
    plt.ioff()

thiscnt = 0
#0-20    20-25 1*loss_cl 6 8     25-  10*loss_cl  

writer = SummaryWriter('./log/%s'%(str(datetime.date.today())))


for epoch in range(0,opt.nepoch+1):
    gen.train()
    
    for i,data in enumerate(traindataloader):
        running_loss = 0
        running_loss_cl = 0
        running_lossG1 = 0
        running_lossG2 = 0
        
        img = []
        
        pointcloud_tmp = []
        images, points = data[0],data[1]
        
        #print('shape',images.shape,type(images))
        img = images.cpu().numpy()
        np.save("./result/train_img.npy",img)

        
        points = Variable(points.float())
        points = points.cuda()
        images = Variable(images.float())
        images = images.cuda()
        points = points.unsqueeze(1)
        
        for i_ in range(4):
            for j_ in range(3):
                writer.add_image('image%s%s'%(i_,j_),images[i_][j_])
        #writer.add_graph(gen,(images,))
        #print('shape here ',images.shape,points.shape)
        feature , fake , coarse_point = gen(images)
        coarse_point = coarse_point.transpose(3,2).unsqueeze(2)
        
        fake = fake.unsqueeze(1)
        
        lossG1 = 0
        # # gt fake
        for index_i in range(4):
            for index_j in range(3):
                #print('shape here ',points[index_i].shape, coarse_point[index_i][index_j].shape)
                dist1, dist2 = chamfer_dist(points[index_i], coarse_point[index_i][index_j])
                lossG1 = lossG1 + (torch.mean(dist1)) + (torch.mean(dist2))
        
        for index_ in range(4):         
            dist1_, dist2_ = chamfer_dist_(points[index_], fake[index_])
            lossG2 = (torch.mean(dist1_)) + (torch.mean(dist2_))
        
        # loss1 = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
        # loss2 = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
        # q1 = feature[0][0].unsqueeze(dim=0)
        # positive_key1 = feature[0][1].unsqueeze(dim=0)
        # negative_key1 = torch.stack([feature[1],feature[2],feature[3]],dim=0)
        # negative_key1 = negative_key1.reshape(9,512)
        # #print(q1.shape,positive_key1.shape,negative_key1.shape)
        # output1 = loss1(q1, positive_key1, negative_key1)
        # q2 = feature[0][1].unsqueeze(dim=0)
        # positive_key2 = feature[0][2].unsqueeze(dim=0)
        # output2 = loss2(q2, positive_key2, negative_key1)

        # loss_cl = output1 + output2  
        
        #print('123123',feature.shape,fake.shape,coarse_point.shape,points.shape)
        #torch.Size([4, 3, 512]) torch.Size([4, 1, 1024, 3]) torch.Size([4, 3, 1, 1024, 3]) torch.Size([4, 1, 1024, 3])
        #loss_kl_1 = nn.KLDivLoss(reduction="batchmean",log_target=True)
        # loss_kl_2 = nn.KLDivLoss(reduction="batchmean",log_target=True)
        # loss_kl_3 = nn.KLDivLoss(reduction="batchmean",log_target=True)
        # loss_kl_4 = nn.KLDivLoss(reduction="batchmean",log_target=True)
        
        fake1 = fake.reshape(4,1,3,1024)
        points1 = points.reshape(4,1,3,1024)
        #print('123123',feature.shape,fake1.shape,coarse_point.shape,points1.shape)
        
        loss_KL = 0
        #kl 散度 inc是放大倍数
        
        # inc = 1000
        # for i_index in range(4):
        #     for j_index in range(3):
        #         #print('shape hereeee',(F.log_softmax(inc*fake1[i_index][0][j_index],dim=0)).shape)
        #         #print('value ',F.log_softmax(inc*fake1[i_index][0][j_index],dim=0))
        #         loss_KL  = loss_KL + loss_kl_1((F.log_softmax(inc*fake1[i_index][0][j_index],dim=0)),(F.log_softmax(inc*points1[i_index][0][j_index],dim=0)))
        #print('kl is ',loss_KL)
        # [0,50] 1 6 20  
        lossG =6*lossG1 + 20*lossG2
        running_loss = lossG.item() + running_loss
        #running_loss_cl = loss_cl.item() + running_loss_cl
        running_lossG1 = lossG1.item() + running_lossG1
        running_lossG2 = lossG2.item() + running_lossG2
        optimizerG.zero_grad()
        lossG.backward()  
        optimizerG.step()
        #if i % 100 == 0:
        print('[%d: %d/%d] train lossG: %f' %(epoch, i, num_batch, lossG.item()))
        print('                      ')
        if i % 150==0 :
            torch.save(gen.state_dict(), 'model_%s/model_%d_%d.pth' %(str(opt.cats[0]),epoch,i))         
        else:
            pass 
        if thiscnt %50==0:
            term.append(thiscnt)
            loss_visual.append(running_loss)
            #print('term',term,'loss ',loss_visual)
            draw_loss(term,loss_visual)
            running_loss = 0
            # test per 50 batch to overwatch overfitting
        thiscnt = thiscnt + 1   
    
    cd_epoch,iou_epoch = test_cl_iou(epoch)
    
    writer.add_scalar('loss_all  ', running_loss, global_step=epoch)
    #writer.add_scalar('loss_cl  ', running_loss_cl, global_step=epoch)
    writer.add_scalar('lossG1 ', running_lossG1, global_step=epoch)
    writer.add_scalar('lossG2 ', running_lossG2, global_step=epoch)
    writer.add_scalar('test loss epoch ', iou_epoch, global_step=epoch)
    writer.add_scalar('test_cd_loss  ', cd_epoch, global_step=epoch)
    
    if epoch % 5 == 0 :
            opt.lr = opt.lr * 0.5
            for param_group in optimizerG.param_groups:
                param_group['lr'] = opt.lr
            print('lr decay:', opt.lr)
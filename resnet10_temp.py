#!/usr/bin/env python
# coding: utf-8

#참고한 깃허브 모델
#https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py -> 이 모델에서 block [1,1,1,1] 로만 바꿈.
#https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py -> 이 모델 그냥 가져다씀 모델 구성 완전 같음.
#https://github.com/facebook/fb.resnet.torch -> lua 언어로 되어 있지만 모델 구성 완전 같음
#https://github.com/cvjena/cnn-models/tree/master/ResNet_preact -> 논문에서 참고한 caffe로 만든 resnet10 모델, 하지만 모델 구성이 약간 다름

#class 별 최소 데이터 개수 589개 (label=242)
#1300개가 아닌 class 개수 = 106
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data.dataset import Dataset

import os
import argparse
import sys
import shutil

import visdom
import random
import numpy as np
import time
import xlrd
import glob
import os
import pandas as pd
from scipy.misc import imread, imsave, imresize
from PIL import Image



# In[2]:


parser = argparse.ArgumentParser(description = 'RESNET10')

#train or test
parser.add_argument('--istrain', type=int, default=1, help='train = 1, test = 0')

#epoch, batch_size, num_workers(프로세스 수), lr, fold
parser.add_argument('--epochs', type=int, default=50, help='epoch')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--workers', type=int, default=4, help='workers')
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
parser.add_argument('--folds', type=int, default=10, help='folds')

#user parameter
parser.add_argument('--majority', type=int, default=1300, help='majority class count')
parser.add_argument('--minority', type=int, default=500, help='minority class count')
parser.add_argument('--ratio', type=int, default=10, help='ratio between majority class count and minority count')

args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, xlsx_path, transforms=None):
        """
        Args:
            xlsx_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        total_class = 1000
        
        # 여기서 이제 사용자 입력한 majority, minority, ratio 작업 합니다.
        self.majority_count = int(total_class*(1/(args.ratio+1)))
        self.minority_count = total_class-self.majority_count
        
        #transforms
        if transforms is not None:
            self.transforms = transforms
        # First column contains the image paths
        self.image_arr = []
        # Second column is the labels
        self.label_arr = []
        
        #
        self.imbalance_image_arr = []
        self.imbalance_data_arr = []
        # 각 데이터당 전체 개수 조정
        self.data_count = []
        
        self.pre_p()
        self.post_p()
        
        # Calculate len
        self.data_len = len(self.imbalance_image_arr)
        
        self.showinfo()

    def __getitem__(self, index):
        # Get image name from the pandas df
        img = self.imbalance_image_arr[index]

        # Transform image to tensor
        img_as_tensor = self.transforms(img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.imbalance_label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
    def pre_p(self):
        for i in range(1, 1001):
            data_path = 'Y:/ILSVRC2012_train/'+str(i)
            img_list = os.listdir(data_path)
            
            self.data_count.append(len(img_list))
            
            for img_path in img_list:
                path = data_path + '/' + img_path
                img_as_img = Image.open(path)
                img_as_img = img_as_img.convert("RGB")
                self.image_arr.append(img_as_img)
                self.label_arr.append(i)
                
    def post_p(self):
        
        total_class = 1000
        majority_list = []
        minority_list = []
        
        #각 데이터 몇개 인지 조사해 1300개 미만인 것은 minority에 추가하기
        for i in range(0,1000):
            if data_count[i] <1300:
                minority_list.append(i+1)
                
        #데이터 imbalance 작업
        for i in range(0, self.majority_count):
    
            while rand_num in majority_list or rand_num in minority_list:
                rand_num = random.randint(1,total_class)
                
            majority_list.append(rand_num)
        majority_list.sort()
        
        for i in range(1, 1001):
            if i not in majority_list and i not in minority_list:
                minority_list.append(i)
        minority_list.sort()
        
        ###
        for major in majority_list:
            
            cur_major = []
            for i in range((major-1)*data_count[major-1],(major)*data_count[major-1]):
                if self.data_info['label'][i] == major:
                    cur_major.append(i)
            
            live_list = []
            rand_num = random.randint(cur_major[0],cur_major[(data_count[major-1]-1)])
            for i in range(0, args.majority):
                while rand_num in live_list:
                    rand_num = random.randint(cur_major[0],cur_major[(data_count[major-1]-1)])
                live_list.append(rand_num)
            live_list.sort()
            
            for index in live_list:
                self.imbalance_image_arr.append(self.image_arr[index])
                self.imbalance_label_arr.append(self.label_arr[index])
        print('majority finish')
        
        for minor in minority_list:
            cur_minor = []
            for i in range((minor-1)*data_count[minor-1],(minor)*data_count[minor-1]):
                if self.data_info['label'][i] == minor:
                    cur_minor.append(i)
                    
            live_list = []
            rand_num = random.randint(cur_minor[0],cur_minor[(data_count[minor-1]-1)])
            for i in range(0, args.minority):
                while rand_num in live_list:
                    rand_num = random.randint(cur_minor[0],cur_minor[(data_count[minor-1]-1)])
                live_list.append(rand_num)
            live_list.sort()
            
            for index in live_list:
                self.imbalance_image_arr.append(self.image_arr[index])
                self.imbalance_label_arr.append(self.label_arr[index])
        print('miniority finish')
        
    def showinfo(self):
        print('total data count: ', self.data_len)
        print('majority classes: ', args.majority)
        print('minority classes: ', args.minority)
        print('ratio: ', args.ratio)
        print('total class: ', 1000)
        print('classes per majority class: ', self.majority_count)
        print('classes per minority class: ', self.minority_count)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.stride = stride
        self.downsample = downsample

        #Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # if self.stride == 1:
        #     # 1 layer 층
        #     self.bn1 = norm_layer(inplanes)
        #     self.conv1 = conv3x3(inplanes, planes)
        #     self.relu = nn.ReLU(inplace=True)
        #     self.bn2 = norm_layer(planes)
        #     self.conv2 = conv3x3(planes, planes)
        #     self.relu_last = nn.ReLU(inplace=True)
        #
        # # 2, 3, 4 layer 층
        # elif self.stride == 2:
        #     self.conv0 = conv1x1(inplanes, planes, self.stride)
        #     self.bn1 = norm_layer(inplanes)
        #     self.conv1 = conv3x3(inplanes, planes, self.stride)
        #     self.relu = nn.ReLU(inplace=True)
        #     self.bn2 = norm_layer(planes)
        #     self.conv2 = conv3x3(planes, planes)
        #     self.relu_last = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        
        return out
        # if self.stride==1:
        #     identity = x
        #     out = self.bn1(x)
        #     out = self.conv1(out)
        #     out = self.relu(out)
        #     out = self.bn2(out)
        #     out = self.conv2(out)
        #     out = self.relu_last(out)
        #     if self.downsample is not None:
        #         identity = self.downsample(x)
        #     out = out + identity
        #     return out
        #
        # elif self.stride==2:
        #     identity = x
        #     identity = self.conv0(identity)
        #     out = self.bn1(x)
        #     out = self.conv1(out)
        #     out = self.relu(out)
        #     out = self.bn2(out)
        #     out = self.conv2(out)
        #     out = self.relu_last(out)
        #     if self.downsample is not None:
        #         identity = self.downsample(x)
        #     out = out + identity
        #     return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet10(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def SaveModel(fold, epoch, step):
    print("Save model")
    directory_path = './model_backup/'+str(fold)+'_'+str(args.majority)+'_'+str(args.minority)+'_'+str(args.ratio)+'/'
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    save_path = './model_backup/'+str(fold)+'_'+str(args.majority)+'_'+str(args.minority)+'_'+str(args.ratio)+'/'+str(epoch).zfill(5)+'_'+str(step).zfill(5)+'.ckpt'
    torch.save(model.state_dict(), save_path)

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #num_classes = 1000, 어차피 1000개
    #learning_rate = 0.001
    
    #train or test select
    if args.istrain==1:
        for fold in range(args.folds):
            #데이터 비율 조절 필요하다

            #train loader
            transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_set = CustomDataset('./validation_file_list.xlsx',transform)
            #train_set = datasets.ImageFolder('Y:\\ILSVRC2012\\train',transform)
            train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=args.workers)

            #model
            model = resnet10().to(device)

            #Loss and Optimizer
            criterion = nn.CrossEntropyLoss() # regression은 MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            #visdom
            vis = visdom.Visdom()
            plot = vis.line(Y=torch.zeros([1,1], dtype=torch.float64), X=np.array([0]))

            # Train the model
            print('train start!')
            total_step = len(train_loader)
            for epoch in range(args.epochs):
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    #loss = criterion(outputs.float(), labels.float()) -> regression?
                    plot = vis.line(Y=np.array([loss.item()]), X=np.array([epoch * train_loader.__len__() + i + 1]), win=plot, update='append')

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % 100 == 0:
                        print ('Fold [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                                .format(fold+1 ,args.folds, epoch+1, args.epochs, i+1, total_step, loss.item()))

                        #print(outputs.float(), labels.float())

                        # Save the model checkpoint, batch가 16이니 1600 마다 checkpoint 저장한다
                if (epoch+1) % 1 == 0:
                    SaveModel(fold+1, epoch+1, i+1)
    else:
        
        #test loader
        transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = datasets.ImageFolder('Y:\\ILSVRC2012\\test',transform)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                           batch_size=args.batch_size, 
                                           shuffle=False, 
                                           num_workers=args.workers)
        
                
        highest_model_p = ''
        highest_accuracy = 0
        for fold in range(args.folds):
            
            load_path = './model_backup/'+str(fold+1)+'_'+str(args.majority)+'_'+str(args.minority)+'_'+str(args.ratio)

            #model_list - checkpoint 모델 전부 성능 편가
            model_list = os.listdir(load_path)
            model_list.sort()

            print('test start!')

            for (i,model_p) in enumerate(model_list):
                
                model = resnet10().to(device)
                model_path = load_path+'/'+model_p
                if i ==0 && fold ==1:
                    highest_model_p = model_path
                model.load_state_dict(torch.load(model_path))

                model.eval()

                print(model_p+' is tested..')

                with torch.no_grad():
                    correct = 0
                    total = 0

                    for i,(images, labels) in enumerate(test_loader):
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()


                    accuracy_temp = 100 * correct/total
                    print('Accuracy of test images : %d %%'% (accuracy_temp))

                    if highest_accuracy < accuracy_temp:
                        highest_accuracy = accuracy_temp
                        highest_model_p = model_path

                    print('current highest accuracy of model')
                    print(highest_model_p)
                    print(highest_accuracy)

            print('highset accuracy of models')
            print(highest_model_p)
            print(highest_accuracy)
            print()

        high_save_path = './model_backup/highest/'
        if not os.path.isdir(high_save_path):
            os.mkdir(high_save_path)
        file_path = highest_model_p
        shutil.copy(file_path, high_save_path)


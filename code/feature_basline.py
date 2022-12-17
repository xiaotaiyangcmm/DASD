#每张图片提取特征
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

from model_modify_baseline import resnet50,randompinjie
from tsn_utils import TSN_yuchuli

prediction=[]
def resnet_feature(net,x):
    x = randompinjie(x,'val')
    
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)

    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)

    if net.include_top:
        x = net.avgpool(x)
        x = torch.flatten(x, 1)

    return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_feature(weights_path,csv_path):
    classes = ('abnormal', 'normal')
    model = resnet50(num_classes=2).to(device)

    # model = nn.DataParallel(model)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    img=TSN_yuchuli(csv_path,32,4,'middle','sample',0.1)
    feature=torch.zeros(4,2048)

    model.eval()
    with torch.no_grad():
        for i in range(4):
            x = resnet_feature(model,img[i].unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device))
            output = torch.squeeze(model(img[i].unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device),'val'))
            predict = torch.max(output.reshape(1,2), dim=1)[1]
            # print(classes[predict])
            prediction.append([classes[predict]])
            feature[i] = x 

    return feature

#ASD数据集

path='/data/caimiaomiao/ASD2/allvideo/5flod/flod3/subset3/abnormal'
savepath='/data/caimiaomiao/ASD2/allvideo/8flod_resnet50_feature/flod3'
weights_path = "/data/caimiaomiao/ASD2/100clip_one-one/resent_train/Model_parameters_cross/finnal/100_resnet50_baseline/model-best-weight3.pth"
mynum=100
dirs = os.listdir(path) 
dirs.sort()
for i in range(len(dirs)):                             # 循环读取路径下的文件并筛选输出   
    print(os.path.join(path,dirs[i]))
    if i==0:
        feature=get_feature(weights_path ,os.path.join(path,dirs[i]))
        feature=feature.cpu().detach().numpy()
    else:
        x=get_feature(weights_path ,os.path.join(path,dirs[i]))
        x=x.cpu().detach().numpy()
        feature = np.append(feature,x,axis=0)
# np.savetxt(savepath+'/subset1_abnormal.csv', feature, delimiter = ',')     


test=pd.DataFrame({'prediction':prediction})
print(test)
test.to_csv(savepath+'/index-abnormal.csv',encoding='gbk')
#测试结果 cell1-3
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_modify import resnet50
from tsn_utils import TSN_yuchuli
from collections import Counter
import random
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import json
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="ASD")
    parser.add_argument('--rootpath', type=str,default='/data/caimiaomiao/ASD2/GitHub', help='rootpath')
    parser.add_argument('--mynum', type=int, default=100, help='number of head-related features')
    parser.add_argument('--numsegments', type=int, default=4, help='number of segmentation')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--epochs', type=int, default=10000, help='epochs')
    parser.add_argument('--model_name',default='attention' ,choices=['attention','baseline'])
    args = parser.parse_args()
    return args
    
def predict(img_path, weights_path,mynum,myname,num_segment):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = TSN_yuchuli(img_path,32,num_segment,'middle','sample',0.1)

    # create model
    model = resnet50(num_classes=2,select_num=mynum,select_name=myname).to(device)
    model = nn.DataParallel(model)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        val_images= torch.stack(img, 0).unsqueeze(1).type(torch.FloatTensor)
        outputs = model(val_images.to(device),'val').mean(0).reshape(1,2) 
            
        predict = torch.max(outputs, dim=1)[1]
        print(predict)
    return  predict


def chuanshu(dataset,classs,weights_path,mynum,myname,num_segment):
    FfilePath =dataset+'/'+classs
    bigname=os.listdir(FfilePath)
    bigname.sort() 
    result=np.zeros((len(bigname))) 
    for ind in range(len(bigname)):
        Ffilr=os.path.join( FfilePath, bigname[ind])
        result[ind]=predict(Ffilr,weights_path,mynum,myname,num_segment)
    return result

def jisuan(dataset,weights_path,num_segment,mynum):
    myname='single'

    result_normal=chuanshu(dataset,"normal",weights_path,mynum,myname,num_segment)
    cou_normal=Counter(result_normal)
    print('normal={}'.format(cou_normal[1.0]/result_normal.shape[0]))

    result_abnormal=chuanshu(dataset,"abnormal",weights_path,mynum,myname,num_segment)
    cou_abnormal=Counter(result_abnormal)
    print('abnormal={}'.format(cou_abnormal[0.0]/result_abnormal.shape[0]))

    print('all={}'.format((cou_normal[1.0]+cou_abnormal[0.0])/(result_normal.shape[0]+result_abnormal.shape[0])))
    return result_normal,result_abnormal

def jieguo(result_normal,result_abnormal):
    gt=np.append(result_normal,result_abnormal)
    pred=np.append(np.ones(result_normal.shape[0]),np.zeros(result_abnormal.shape[0]))

    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    auc = roc_auc_score(gt,pred)
    Acc=(TP+TN)/float(TP+TN+FP+FN)
    f=f1_score(gt,pred)
    Spe=TN / float(TN+FP)
    Sen=TP / float(TP+FN)
    return auc,Acc,f,Spe,Sen


def main():
    args = get_args()
    datasetpath=args.rootpath+'/dataset/test'
    namee='attention_{}_segmentation_{}'.format(args.numsegments,args.mynum)
    weights_path=args.rootpath+'/outputs/'+namee+'/'+'weight-model-best.pth'
    result_normal1,result_abnormal1=jisuan(datasetpath,weights_path,args.numsegments,args.mynum)
    auc1,Acc1,f1,Spe1,Sen1=jieguo(result_normal1,result_abnormal1)
    jsondata1={'Accuracy':Acc1,'f1_score':f1,'Specificity':Spe1,'Sensitivity':Sen1,'AUC':auc1}

    with open(os.path.dirname(weights_path)+'/results.json', 'w') as result_file:
        json.dump(jsondata1, result_file)
        
if __name__ == "__main__":
    main()
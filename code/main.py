import os
import json
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import os, sys
import pandas as pd
import random
import numpy
from tsn_utils import read_split_data
from tsn_my_dataset import MyDataSet
import torch.nn.functional
from mytransformer import RandomResizedCrop
import torch.nn.functional as F
import argparse
import csv
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

def main():
    args = get_args()
    os.chdir(args.rootpath+'/code')
    sys.path.append(os.getcwd())
    modelname=args.model_name
    if modelname=='attention':
        from model_modify import resnet50
    elif modelname=='baseline':
        from model_modify_baseline import resnet50
    ###################################################################################################################
    #attention
    mynum=args.mynum
    myname='single'
    ###################################################################################################################
    batch_size =args.batch_size
    val_batch_size =1
    epochs =args.epochs
    lr=0.001
    ###################################################################################################################
    #TSN
    window_ratio=0.1
    framesperclip = 32
    numsegments = args.numsegments
    jittermode=2
    ###################################################################################################################
    #flod
    cross_title="train"
    val_file_name='test'
    weightname="weight1"
    ###################################################################################################################
    namee='attention_{}_segmentation_{}'.format(numsegments,mynum)

    datasetpath=args.rootpath+'/dataset'
    train_file_name='train'
    savepath=args.rootpath+'/outputs/'+namee
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    root = datasetpath  # flower data set path         
    ###################################################################################################################
    def seed_torch(seed=1024):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch()
    seed=1024
    def _init_fn(worker_id):
        np.random.seed(int(seed)+worker_id)
    ###################################################################################################################
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    ###################################################################################################################
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root,train_file_name,val_file_name)
    weight=torch.tensor([1,train_images_label.count(0)/train_images_label.count(1)]).type(torch.FloatTensor)
    train_dataset = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            frames_per_clip=framesperclip,
                            num_segments=numsegments,
                            jitter_mode=jittermode,
                            name='sample')
    train_num = len(train_dataset)
    ###################################################################################################################
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    ###################################################################################################################
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
    #                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=_init_fn)
    validate_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                frames_per_clip=framesperclip,
                                num_segments=numsegments,
                                jitter_mode='middle',
                                name='sample',
                                ratio=window_ratio)

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                            batch_size=val_batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
    #                                            collate_fn=validate_dataset.collate_fn,
                                            worker_init_fn=_init_fn)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                        val_num))
    ###################################################################################################################
    if modelname=='attention':
        model = resnet50(num_classes=2,select_num=mynum,select_name=myname).to(device)
    elif modelname=='baseline':
        model = resnet50(num_classes=2).to(device)
    ###################################################################################################################
    gpus = [0,1]
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    ###################################################################################################################
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr)
    ###################################################################################################################
    best_acc = 0.0
    bestaccuu = 0.0
    valaccuu= 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    ###################################################################################################################
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        accu_num=0
        for step, data in enumerate(train_bar):
            img, labels, paths = data
            
            optimizer.zero_grad()
            #softmax################################################################
            for mm in range(len(img)):
                if mm==0:
                    images=img[mm].unsqueeze(1).type(torch.FloatTensor)
                    a = model(images.to(device),'train')
                    a = torch.unsqueeze(a, dim=0)
                else:
                    images=img[mm].unsqueeze(1).type(torch.FloatTensor)
                    c=model(images.to(device),'train')
                    c=torch.unsqueeze(c, dim=0)
                    a=torch.cat([a,c],dim=0)

            logits=a.mean(dim=0)
            #############################################################################
            pred_classes = torch.max(logits, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum().item()
            
            loss = F.cross_entropy(logits, labels.to(device),weight.to(device))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                    epochs,
                                                                    loss)
            
            
            

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_running_loss= 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_img, val_labels, paths = val_data

                ##################################################################
                val_images= torch.stack(val_img, 0).type(torch.FloatTensor)
                outputs = model(val_images.to(device),'val').mean(0).reshape(1,2)          
                #############################################################################
                val_loss = F.cross_entropy(outputs.to(device), val_labels.to(device),weight.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y.to(device), val_labels.to(device)).sum().item()
                val_running_loss += val_loss.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                        epochs)
                
        print('[epoch %d] train_loss: %.3f  train_accu: %.3f  val_loss: %.3f val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, accu_num / train_num ,val_running_loss / val_steps , acc / val_num))

        trainloss= running_loss / train_steps
        valaccuu=acc / val_num
        val_acc=acc / val_num

        if val_acc > best_acc:
            best_acc = val_acc
            ppath=savepath+'/'+weightname+'-model-best.pth'
            torch.save(model.state_dict(), ppath)
            bestaccuu=best_acc

if __name__ == "__main__":
    main()
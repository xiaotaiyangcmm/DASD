import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy as np
import torch
from torchvision import transforms


def read_split_data(root: str,tranname:str,valname:str):
    class_root=os.path.join(root,valname)

    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG",".csv"]  # 支持的文件后缀类型

    setname=[tranname,valname]

    # 遍历每个文件夹下的文件
    for name in setname:
        for cla in flower_class:
            cla_path = os.path.join(root, name,cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(root, name,cla, i) for i in os.listdir(cla_path)
                    if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            if cla==flower_class[0]:
                image_class = 0
            else:
                image_class = 1

            for img_path in images:
                if name == valname:  # 如果该路径在采样的验证集样本中则存入验证集
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:  # 否则存入训练集
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)
    print("{} images were found in the dataset.".format(len(train_images_path)+len(val_images_path)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    


    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        sample_num += images.shape[0]

        c=csvmat(paths)
        pred = model((c.float()).to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def get_chunks(num_vectors, frames_per_clip, num_chunks):
    num_frames = num_vectors - frames_per_clip + 1  # frames covered by the feature vectors
    all_indices = np.arange(num_frames)
    chunks = np.array_split(all_indices, num_chunks)
    return chunks

def get_jittered_indices(chunks, frames_per_clip, mode, seizure_dir=None):
        if mode == 1:
            mode = 'uniform'
        elif mode == 'delta':
            mode = 'middle'
        else:
            try:
                float(mode)
                concentration = mode
                mode = 'beta'
            except ValueError:  # probably 'normal', left for bw compatibility
                pass


        jittered_indices = []
        for indices_in_chunk in chunks:
            if len(indices_in_chunk) <= frames_per_clip:
                message = (
                    f'Chunk length ({len(indices_in_chunk)}) is less than'
                    f' the number of frames per snippet ({frames_per_clip})'
                )
                if seizure_dir is not None:
                    message += f'. Seizure dir: {seizure_dir}'
                raise RuntimeError(message)
            first_possible = int(indices_in_chunk[0])
            last_possible = int(indices_in_chunk[-1] - frames_per_clip + 1)
            if mode == 'uniform':
                index = torch.randint(first_possible, last_possible + 1, (1,))
                index = index.item()
            elif mode == 'normal':
                mean = (last_possible + first_possible) / 2
                indices_range = last_possible - first_possible
                # Values less than three standard deviations from the mean
                # account for 99.73% of the total
                std = indices_range / 6
                index = mean + std * torch.randn(1)
                # We still need to clip for that 0.27%
                index = index.round()
                index = torch.clamp(index, first_possible, last_possible).int()
                index = index.item()
            elif mode == 'middle':
                index = int(round((last_possible + first_possible) / 2))
            elif mode == 'beta':
                m = torch.distributions.beta.Beta(
                    concentration,
                    concentration,
                )
                sample = m.sample()
                sample *= last_possible - first_possible
                sample += first_possible
                index = sample.round().long().item()
            jittered_indices.append(index)
        return jittered_indices

def TSN_yuchuli(seizure_dir,frames_per_clip,num_segments,jitter_mode,name,ratio=0.1):

    df = pd.read_csv(seizure_dir)
    df=df[df.columns[5:]]
    df=np.array(df)
    num_vectors = df.shape[0]
    if name=='sample': 
        chunks = get_chunks(
                num_vectors,
                frames_per_clip,
                num_segments,
            )#返回数组[ind]，（帧数+frames_per_clip-1）按照num_segments进度分组（平均分成32组）

        jittered_indices = get_jittered_indices(
                    chunks,
                    frames_per_clip,
                    jitter_mode,
                    seizure_dir=seizure_dir,
                )#返回提取视频的第一帧坐标

    if name=='slide':
        window=frames_per_clip
        stride=int((1-ratio)*window)
        jittered_indices=np.arange(0,num_vectors-frames_per_clip+1,stride)

    my_image=list()
    for i in jittered_indices:
        rows=df[np.arange(i,i+frames_per_clip)]
        rows=np.array(rows)
        img=(rows-np.min(rows))/(np.max(rows)-np.min(rows))
        img=transforms.ToTensor()(img)
        my_image.extend(img)

    return my_image
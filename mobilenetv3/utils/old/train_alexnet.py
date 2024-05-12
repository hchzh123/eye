#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cls_torch_tem 
@File    ：train_alexnet.py
@Author  ：ChenmingSong
@Date    ：2022/3/29 10:14 
@Description：
'''
import torch
from torch import nn
from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

data_path = "../../../flowers_data"
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    "img_size": 224,  # 图片输入大小
    "train_dir": osp.join(data_path, "train"),  # todo 训练集路径
    "val_dir": osp.join(data_path, "val"),  # todo 验证集路径
    'device': device,  # 设备
    'lr': 1e-4,  # 学习率
    'batch_size': 4,  # 批次大小
    'num_workers': 0,  # 进程
    'epochs': 10,  # 轮数
    "save_dir": "../checkpoints/",  # todo 保存路径
    "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    'weight_decay': 1e-5  # 学习率衰减
}


# 定义模型
def get_alexnet(num_classes):
    net = nn.Sequential(
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 2048), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 512), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(512, num_classes))
    return net


# 定义训练流程
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):  # 开始训练
        images = images.to(params['device'], non_blocking=True)  # 加载数据
        target = target.to(params['device'], non_blocking=True)  # 加载模型
        output = model(images)  # 数据送入模型进行前向传播
        loss = criterion(output, target.long())  # 计算损失
        f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
        recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
        acc = accuracy(output, target)  # 计算准确率分数
        metric_monitor.update('Loss', loss.item())  # 更新损失
        metric_monitor.update('F1', f1_macro)  # 更新f1
        metric_monitor.update('Recall', recall_macro)  # 更新recall
        metric_monitor.update('Accuracy', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新优化器
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]  # 返回结果


# 定义验证流程
# todo 想办法写一个验证的，把验证的流程走一下，计算出具体的实验指标出来。
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            loss = criterion(output, target.long())  # 计算损失
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]


def show_loss_acc(acc, loss, val_acc, val_loss):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('record/results_alexnet.png', dpi=100)

# 训练过程中得折线图
# 最后验证的流程图
if __name__ == '__main__':
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)  # 加载训练集
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    save_dir = params['save_dir']  # 设置模型保存路径
    if not osp.isdir(save_dir):  # 如果保存路径不存在的话就创建
        os.makedirs(save_dir)  #
        print("{} created".format(save_dir))
    train_loader = DataLoader(  # 按照批次加载训练集
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )
    print(train_dataset.classes)
    model = get_alexnet(params["num_classes"])
    print(model)
    # model = nn.DataParallel(model)  # 模型并行化，提高模型的速度
    model = model.to(params['device'])  # 模型部署到设备上
    criterion = nn.CrossEntropyLoss().to(params['device'])  # 设置损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  # 设置优化器
    best_acc = 0.0  # 记录最好的准确率
    # 只保存最好的那个模型。
    for epoch in range(1, params['epochs'] + 1):  # 开始训练
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        if acc >= best_acc:
            save_path = osp.join(save_dir, f"Alexnet_{epoch}epochs_accuracy{acc:.5f}_weights.pth")
            torch.save(model.state_dict(), save_path)
            best_acc = acc
    show_loss_acc(accs, losss, val_accs, val_losss)
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
from train_alexnet import get_alexnet

data_path = "../../../flowers_data"
model_path = "../checkpoints/Alexnet_7epochs_accuracy0.73390_weights.pth"  # todo
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    "img_size": 224,  # 图片输入大小
    "test_dir": osp.join(data_path, "test"),  # todo 验证集路径
    'device': device,  # 设备
    'batch_size': 4,  # 批次大小
    'num_workers': 0,  # 进程
    "num_classes": len(os.listdir(osp.join(data_path, "test"))),  # 类别数目, 自适应获取类别数目
}


# 定义验证流程
def test(val_loader, model, params, class_names):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            # loss = criterion(output, target.long())  # 计算损失
            # print(output)
            target_numpy = target.cpu().numpy()
            y_pred = torch.softmax(output, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            test_real_labels.extend(target_numpy)
            test_pre_labels.extend(y_pred)
            # print(target_numpy)
            # print(y_pred)
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            # metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "mode: {epoch}.  {metric_monitor}".format(
                    epoch="test",
                    metric_monitor=metric_monitor)
            )

    # for test_batch_images, test_batch_labels in test_ds:
    #     test_batch_labels = test_batch_labels.numpy()
    #     test_batch_pres = model.predict(test_batch_images)
    #     # print(test_batch_pres)
    #
    #     test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
    #     test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
    #     # print(test_batch_labels_max)
    #     # print(test_batch_pres_max)
    #     # 将推理对应的标签取出
    #     for i in test_batch_labels_max:
    #         test_real_labels.append(i)
    #
    #     for i in test_batch_pres_max:
    #         test_pre_labels.append(i)
    # break

    # print(test_real_labels)
    # print(test_pre_labels)
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    # print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    # print()
    heat_maps_float = heat_maps / heat_maps_sum
    # print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="record/heatmap_alexnet.png")

    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['F1']["avg"], \
           metric_monitor.metrics['Recall']["avg"]


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    # plt.show()


# 训练过程中得折线图
if __name__ == '__main__':
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    test_dataset = datasets.ImageFolder(params["test_dir"], valid_transforms)  # 加载训练集
    class_names = test_dataset.classes
    val_loader = DataLoader(  # 按照批次加载验证集
        test_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )
    model = get_alexnet(params["num_classes"])
    # model = nn.DataParallel(model)  # 模型并行化，提高模型的速度
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model = model.to(params['device'])  # 模型部署到设备上

    # 只保存最好的那个模型。
    acc, f1, recall = test(val_loader, model, params, class_names)
    print("测试结果：")
    print(f"acc: {acc}, F1: {f1}, recall: {recall}")

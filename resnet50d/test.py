
# -*- coding: UTF-8 -*-

from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
from train import SELFMODEL

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

data_path = " "
model_path = " "
model_name = 'resnet50d'
img_size = 224
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    # 'model': 'vit_tiny_patch16_224',  #
    # 'model': 'efficientnet_b3a',
    'model': model_name,
    "img_size": img_size,
    "test_dir": osp.join(data_path, "test"),
    'device': device,
    'batch_size': 4,
    'num_workers': 0,
    "num_classes": len(os.listdir(osp.join(data_path, "train"))),
}


def mytest(val_loader, model, params, class_names):
    print('sb')
    print(val_loader)
    print('hhhh')
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)


    test_real_labels = []
    test_pre_labels = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True)
            output = model(images)  # 前向传播

            # print(output)
            target_numpy = target.cpu().numpy()
            y_pred = torch.softmax(output, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            test_real_labels.extend(target_numpy)
            test_pre_labels.extend(y_pred)
            # print(target_numpy)
            # print(y_pred)
            f1_macro = calculate_f1_macro(output, target)
            recall_macro = calculate_recall_macro(output, target)
            acc = accuracy(output, target)

            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "mode: {epoch}.  {metric_monitor}".format(
                    epoch="test",
                    metric_monitor=metric_monitor)
            )
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
                  save_name="record/heatmap_{}.png".format(model_name))
    # 加上模型名称

    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['F1']["avg"], \
           metric_monitor.metrics['Recall']["avg"]


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):

    fig, ax = plt.subplots()

    im = ax.imshow(harvest, cmap="OrRd")

    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))

    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


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


if __name__ == '__main__':
    data_transforms = get_torch_transforms(img_size=params["img_size"])

    valid_transforms = data_transforms['val']

    test_dataset = datasets.ImageFolder(params["test_dir"], valid_transforms)
    class_names = test_dataset.classes
    print(class_names)

    test_loader = DataLoader(
        test_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )

    # 加载模型
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],
                      pretrained=False)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    tesr =test_loader
    acc, f1, recall = mytest(tesr, model, params, class_names)
    print("测试结果：")
    print(f"acc: {acc}, F1: {f1}, recall: {recall}")
    print("测试完成，heatmap保存在{}下".format("record"))

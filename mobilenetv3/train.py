from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
import csv
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
data_path = ''

params = {#mobilenetv3_small_100
    'model': 'mobilenetv3_small_100',
    "img_size": 224,
    "train_dir": osp.join(data_path, "train"),
    "val_dir": osp.join(data_path, "val"),
    'device': device,
    'lr': 1e-3,
    'batch_size': 4,
    'num_workers': 0,
    'epochs': 200,
    "save_dir": "../checkpoints/",
    "pretrained": True,
     "num_classes": len(os.listdir(osp.join(data_path, "train"))),
    'weight_decay': 1e-5
}


# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if model_name[:3] == "res":
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, out_features)
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, out_features)
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)

        print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


# 定义训练流程
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    nBatch = len(train_loader)
    stream = tqdm(train_loader)

    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params['device'], non_blocking=True)
        target = target.to(params['device'], non_blocking=True)
        output = model(images)
        loss = criterion(output, target.long())
        f1_macro = calculate_f1_macro(output, target)
        recall_macro = calculate_recall_macro(output, target)
        acc = accuracy(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('F1', f1_macro)
        metric_monitor.update('Recall', recall_macro)
        metric_monitor.update('Accuracy', acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )

    # Save metrics to CSV file
    with open('training_log.csv', mode='a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Accuracy', 'Loss', 'F1', 'Recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if epoch == 1:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'Accuracy': metric_monitor.metrics['Accuracy']["avg"],
            'Loss': metric_monitor.metrics['Loss']["avg"],
            'F1': metric_monitor.metrics['F1']["avg"],
            'Recall': metric_monitor.metrics['Recall']["avg"]
        })

    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]

# 定义验证流程
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)

    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True)
            output = model(images)
            loss = criterion(output, target.long())
            f1_macro = calculate_f1_macro(output, target)
            recall_macro = calculate_recall_macro(output, target)
            acc = accuracy(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )

    # Save metrics to CSV file
    with open('validation_log.csv', mode='a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Accuracy', 'Loss', 'F1', 'Recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if epoch == 1:
            writer.writeheader()

        writer.writerow({
            'Epoch': epoch,
            'Accuracy': metric_monitor.metrics['Accuracy']["avg"],
            'Loss': metric_monitor.metrics['Loss']["avg"],
            'F1': metric_monitor.metrics['F1']["avg"],
            'Recall': metric_monitor.metrics['Recall']["avg"]
        })

    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]



def show_loss_acc(acc, loss, val_acc, val_loss, sava_dir):

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

    save_path = osp.join(save_dir, "results.png")
    plt.savefig(save_path, dpi=100)


if __name__ == '__main__':
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    data_transforms = get_torch_transforms(img_size=params["img_size"])
    train_transforms = data_transforms['train']
    valid_transforms = data_transforms['val']
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)

    if params['pretrained'] == True:
        save_dir = osp.join(params['save_dir'], params['model']+"_pretrained_" + str(params["img_size"]))
    else:
        save_dir = osp.join(params['save_dir'], params['model'] + "_nopretrained_" + str(params["img_size"]))

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
        print("save dir {} created".format(save_dir))

    train_loader = DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )

    print(train_dataset.classes)
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],
                      pretrained=params['pretrained'])
    model = model.to(params['device'])
    criterion = nn.CrossEntropyLoss().to(params['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_acc = 0.0

    for epoch in range(1, params['epochs'] + 1):
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)

        if val_acc >= best_acc:
            save_path = osp.join(save_dir, f"{params['model']}_{epoch}epochs_accuracy{acc:.5f}_weights.pth")
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc

        # Save metrics to CSV file
        with open('metrics_log.csv', mode='a', newline='') as csvfile:
            fieldnames = ['Epoch', 'Training Accuracy', 'Training Loss', 'Validation Accuracy', 'Validation Loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if epoch == 1:
                writer.writeheader()

            writer.writerow({
                'Epoch': epoch,
                'Training Accuracy': acc,
                'Training Loss': loss,
                'Validation Accuracy': val_acc,
                'Validation Loss': val_loss
            })

    show_loss_acc(accs, losss, val_accs, val_losss, save_dir)
    print("训练已完成，模型和训练日志保存在: {}".format(save_dir))




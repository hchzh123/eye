from m_utils import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# 固定随机种子，保证实验结果是可以复现的
seed = 415
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# 读取数据，这块的操作主要是处理label，把label处理成数字的形式
path = '../data_sd/'  # todo path修改为根路径
labels_file_path = os.path.join(path, 'train.csv')
sample_submission_path = os.path.join(path, 'test.csv')  # todo 如果没有请提前生成测试的文件列表

# 读取训练和测试的csv文件
df = pd.read_csv(labels_file_path)  # 读取训练文件
sub_df = pd.read_csv(sample_submission_path)  # 读取测试文件，也就是最后要提价的文件
labels_unique = df['label'].unique()  # 获取标签

# 把标签处理为数字形式
le = LabelEncoder()
le.fit(df['label'])
df['label'] = le.transform(df['label'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
label_inv_map = {v: k for k, v in label_map.items()}

# 设置超参数
# 先用efficient跑一个基础模型出来看看
params = {
    # 'model': 'seresnext50_32x4d',  # todo 调整预训练模型
    # 'model': 'efficientnet_b3a',  # todo 调整预训练模型
    'model': 'tf_efficientnet_b7_ns',  # todo 调整预训练模型
    # 'model': 'resnet50d',
    'device': device,
    'lr': 1e-3,
    'batch_size': 4,  # todo 调整batchsize大小
    'num_workers': 0,
    'epochs': 50,  # todo 调整
    'out_features': df['label'].nunique(),
    'weight_decay': 1e-5
}


# 训练模型
class LeafNet(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['out_features'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # classifier
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, out_features)
        # n_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(n_features, out_features)
        print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


# 训练流程
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
        acc = accuracy(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('F1', f1_macro)
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
    return metric_monitor.metrics['Accuracy']["avg"]


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
            acc = accuracy(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Accuracy']["avg"]


if __name__ == '__main__':
    # k折交叉验证
    # k折交叉验证的效果不一定好，不如直接在全部数据上进行训练，我是这样感觉的
    # kf = StratifiedKFold(n_splits=1)
    sub_path = "train_images/"  # todo 修改子路径，为了是防止有的图片他路径不对
    test_sub_path = "test_images/"  # todo 修改子路径，为了是防止有的图片他路径不对
    image_col_name = 'image_name'  # todo 修改为csv文件中路径的表头
    label_name = "label"
    # 直接跑一轮就完事了，不需要那么复杂，5折速度太慢了吧也
    # for k, (train_index, test_index) in enumerate(kf.split(df[image_col_name], df[label_name])):
    train_img, valid_img = df[image_col_name], df[image_col_name]
    train_labels, valid_labels = df[label_name], df[label_name]
    train_paths = path + sub_path + train_img
    valid_paths = path + sub_path + valid_img
    test_paths = path + test_sub_path + sub_df[image_col_name]

    train_dataset = LeafDataset(images_filepaths=train_paths.values,
                                labels=train_labels.values,
                                transform=get_train_transforms())
    valid_dataset = LeafDataset(images_filepaths=valid_paths.values,
                                labels=valid_labels.values,
                                transform=get_valid_transforms())
    train_loader = DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )

    val_loader = DataLoader(
        valid_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )
    model = LeafNet()
    model = nn.DataParallel(model)
    model = model.to(params['device'])
    criterion = nn.CrossEntropyLoss().to(params['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    for epoch in range(1, params['epochs'] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        acc = validate(val_loader, model, criterion, epoch, params)
        torch.save(model.state_dict(),
                   f"../checkpoints/{params['model']}_none-flod_{epoch}epochs_accuracy{acc:.5f}_weights.pth")
# 按照这个逻辑来看 k折起作用了沃日

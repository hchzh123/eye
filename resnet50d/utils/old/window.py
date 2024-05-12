# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: ui.py.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from train import SELFMODEL
import numpy as np
from torch import nn
from torchutils import get_torch_transforms
from PIL import Image
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 脑肿瘤切片数据集
# 首先网络结构的部分需要优化一下，然后简化一下，该是什么就是什么，一些简单的结构就不反复赘述了。
# 根据每个花卉写一段特殊的的翻译，然后开始周期性的更新。

class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('花卉识别系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.origin_shape = ()

        self.model_path = "../checkpoints/resnet50d_pretrained_224/resnet50d_10epochs_accuracy0.99501_weights.pth"  # todo  模型路径
        self.classes_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # todo 类名
        self.img_size = 224  # todo 图片大小
        self.model_name = "resnet50d"  # todo 模型名称
        self.num_classes = len(self.classes_names)  # todo 类别数目
        # 加载网络
        model = SELFMODEL(model_name=self.model_name, out_features=self.num_classes, pretrained=False)
        weights = torch.load(self.model_path,
                             map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        model.eval()
        model.to(device)
        self.model = model

        # 加载数据处理
        data_transforms = get_torch_transforms(img_size=self.img_size)
        # train_transforms = data_transforms['train']
        self.valid_transforms = data_transforms['val']
        self.initUI()

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        # mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("识别")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        #
        self.rrr = QLabel("等待识别")
        self.rrr.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.rrr)
        # img_detection_layout.addWidget(self.c1)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        real_about_layout = QVBoxLayout()

        grid_widget = QWidget()
        about_layout = QGridLayout()
        name = QLabel("识别结果")
        age = QLabel("简介")
        time = QLabel("成因")
        suggest = QLabel("防治措施")
        self.result_edit = QLineEdit()
        self.info_edit = QTextEdit()
        self.reason_edit = QTextEdit()
        self.suggest_edit = QTextEdit()

        about_layout.setSpacing(10)
        about_layout.addWidget(name, 1, 0)
        about_layout.addWidget(self.result_edit, 1, 1)

        about_layout.addWidget(age, 2, 0)
        about_layout.addWidget(self.info_edit, 2, 1)

        about_layout.addWidget(time, 3, 0)
        about_layout.addWidget(self.reason_edit, 3, 1)

        about_layout.addWidget(suggest, 4, 0)
        about_layout.addWidget(self.suggest_edit, 4, 1)

        about_title = QLabel('详细结果')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        grid_widget.setFont(font_main)
        grid_widget.setLayout(about_layout)
        real_about_layout.addWidget(about_title)
        real_about_layout.addWidget(grid_widget)
        # real_about_layout.addWidget(go_button)
        about_widget.setLayout(real_about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(about_widget, '详细结果')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
            self.rrr.setText("等待识别")
            self.result_edit.setText("")
            self.info_edit.setText("")
            self.reason_edit.setText("")
            self.suggest_edit.setText("")

    '''
    ***检测图片***
    '''

    def detect_img(self):
        # model = self.model
        # output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = Image.open(source)
        img = self.valid_transforms(img)
        img = img.unsqueeze(0)
        output = self.model(img)
        label_id = torch.argmax(output).item()
        predict_name = self.classes_names[label_id]
        self.rrr.setText("当前识别结果为：{}".format(predict_name))
        # 根据四种疾病显示响应的结果
        # ['myopia', 'normal']
        if predict_name == 'myopia':
            self.result_edit.setText("近视眼")
            self.info_edit.setText("近视是屈光不正的一种。当眼在调节放松状态下，平行光线进入眼内，其聚焦在视网膜之前，这导致视网膜上不能形成清晰像，称为近视眼（myopia）。")
            self.reason_edit.setText(
                "遗传因素\n"
                "大量调查表明，近视具有一定的遗传倾向，常可见家族聚集性，父母双方或一方近视，孩子发生近视的可能性会增大。其中比较明确的是，高度近视的发生为常染色体隐性遗传。"
                "环境因素\n"
                "长期近距离用眼者的近视发生率较高，这也是我国青少年近视高发的主要原因。如果再叠加上环境照明不佳、阅读字迹过小或模糊不清、持续阅读时间过长、缺乏户外活动等因素，更加促使近视的发生与发展。")
            self.suggest_edit.setText("1. 选择正规验光配镜机构咨询，及时科学配镜，矫正视力，防止恶化。"
                                      "\n2. 定期进行眼科检查，包括视力、眼压、视野、眼轴等的变化情况；18岁以下的青少年应该至少每半年进行一次眼科检查。"
                                      "\n3. 保持眼部卫生，不随意揉眼，即使是成年也要注意避免用眼过度疲劳。"
                                      "\n4. 戒烟、少量饮酒的健康生活方式利于视网膜及眼底的保护。"
                                      "\n5. 如果患有糖尿病、高血压等慢性病，一定要定期进行眼部检查，避免眼部并发症。")
        elif predict_name == 'normal':
            self.result_edit.setText("健康")
            self.info_edit.setText("健康")
            self.reason_edit.setText(
                "健康"
            )
            self.suggest_edit.setText(
                "健康")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

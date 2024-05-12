#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cls_template 
@File    ：mv_imgs.py
@Author  ：ChenmingSong
@Date    ：2022/1/6 16:25 
@Description：
'''
import os
import os.path as osp
import pandas as pd
import shutil
def mv_imgs(folder_path="F:/xxxxxxxxxx/sd/smoke/test_images/"):
    df = pd.read_csv("file6143726733.csv")
    image_names = df["image_name"]
    labels = df['label']
    for image_name, label in zip(image_names, labels):
        img_path = osp.join(folder_path, image_name)
        target_folder = osp.join("F:/xxxxxxxxxx/sd/smoke/result/", str(label))
        print(img_path)
        shutil.copy(img_path, target_folder)


if __name__ == '__main__':
    mv_imgs()
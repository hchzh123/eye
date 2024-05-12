#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cls_template_new 
@File    ：img2folder.py
@Author  ：ChenmingSong
@Date    ：2022/2/27 13:24 
@Description：
'''
# https://aistudio.baidu.com/aistudio/datasetdetail/23828/0
# iChallenge-PM中既有病理性近视患者的眼底图片，也有非病理性近视患者的图片，命名规则如下：
# 病理性近视（PM）：文件名以P开头
# 非病理性近视（non-PM）：
# 高度近似（high myopia）：文件名以H开头
# 正常眼睛（normal）：文件名以N开头
# 把图片放在固定的文件夹中
import os
import shutil
import os.path as osp

def move(src_folder, target_folder):
    files = os.listdir(src_folder)
    for file in files:
        file_path = osp.join(src_folder, file)
        file_folder = file[0]
        print(file_folder)
        dst = osp.join(target_folder, file_folder)
        shutil.copy2(file_path, dst)

if __name__ == '__main__':
    move(src_folder="F:/bbbbbbbbbb/tmp/eye/PALM-Training400", target_folder="F:/bbbbbbbbbb/tmp/eye/x")

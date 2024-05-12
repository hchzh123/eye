# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: cls_template
File Name: timm_models.py
Author: chenming
Create Date: 2021/12/18
Description：
-------------------------------------------------
"""
import timm
import torch
# pretrained_resnet_34 = timm.create_model('tf_efficientnet_l2_ns', pretrained=True)
timm_model = timm.create_model('resnest50d', pretrained=True)
# timm_model = timm.create_model('efficientnet_b3a', pretrained=True)
print(timm_model)
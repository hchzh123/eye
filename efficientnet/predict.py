
# -*- coding: UTF-8 -*-

import torch
from train import SELFMODEL
import os
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchutils import get_torch_transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_path = "  "
classes_names = [  ]
img_size = 224
model_name = "efficientnet_b3a"
num_classes = len(classes_names)


def predict_single(model_path, image_path, save_path):
    data_transforms = get_torch_transforms(img_size=img_size)
    valid_transforms = data_transforms['val']


    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)


    img = Image.open(image_path)
    img_original = img.copy()
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    output = model(img)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
    confidence, predicted_class = torch.max(probabilities, 0)

    predict_name = classes_names[predicted_class.item()]
    confidence_str = f"Confidence: {confidence:.2f}%"


    draw = ImageDraw.Draw(img_original)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Predicted: {predict_name}", (255, 255, 255), font=font)
    draw.text((10, 30), confidence_str, (255, 255, 255), font=font)


    img_original.save(save_path)

    print(f"{image_path}'s result is {predict_name} with {confidence_str}. Image saved to {save_path}")


if __name__ == '__main__':
    save_path = "  "
    predict_single(model_path=model_path,
                   image_path=R"  ",
                   save_path=save_path)

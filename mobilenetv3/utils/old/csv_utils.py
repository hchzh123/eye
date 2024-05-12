# 生成csv文件
import os

import pandas as pd


def gen_csv(folder_path, save_name):
    results = []
    image_names = os.listdir(folder_path)
    for image_name in image_names:
        print(image_name)
        result = dict(image_name=image_name)
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_name, index=False)


if __name__ == '__main__':
    gen_csv("../data_sd/test_images", "../data_sd/test.csv")

import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(base_dir, train_dir, val_dir, split_ratio=0.8):
    """
    将图片从 base_dir 划分为训练集 (train_dir) 和验证集 (val_dir)。
    """
    for category in ["cats", "dogs"]:
        category_dir = os.path.join(base_dir, category)
        images = os.listdir(category_dir)

        # 划分数据
        train_images, val_images = train_test_split(images, train_size=split_ratio, random_state=42)

        # 创建目标目录
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

        # 移动图片到训练集和验证集
        for img in train_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(train_dir, category, img))
        for img in val_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(val_dir, category, img))

# 原始数据路径
BASE_DIR = "../data/train"
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/validation"

# 自动划分数据
split_data(BASE_DIR, TRAIN_DIR, VAL_DIR, split_ratio=0.8)
print("训练集和验证集划分完成！")

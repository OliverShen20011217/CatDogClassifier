from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据路径
TRAIN_DIR = "/Users/olivershen/PycharmProjects/CatDogClassifier/data/train"
VALID_DIR = "/Users/olivershen/PycharmProjects/CatDogClassifier/data/test"

# 数据增强（训练集）
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,         # 将像素值归一化到 [0, 1]
    rotation_range=40,         # 随机旋转角度
    width_shift_range=0.2,     # 水平平移
    height_shift_range=0.2,    # 垂直平移
    shear_range=0.2,           # 随机剪切
    zoom_range=0.2,            # 随机缩放
    horizontal_flip=True,      # 随机水平翻转
    fill_mode='nearest'        # 填充模式
)

# 验证集只需要归一化
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 加载训练集
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),    # 调整图片大小
    batch_size=32,             # 批量大小
    class_mode='binary'        # 二分类
)

# 加载验证集
validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(150, 150),    # 调整图片大小
    batch_size=32,             # 批量大小
    class_mode='binary'        # 二分类
)

# 输出数据统计信息
print(f"训练集图片数: {train_generator.samples}")
print(f"验证集图片数: {validation_generator.samples}")

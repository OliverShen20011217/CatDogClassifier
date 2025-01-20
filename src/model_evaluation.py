import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# 设置中文字体（如黑体）
rcParams['font.family'] = font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc').get_name()

# 加载模型
model = load_model('../results/cat_dog_classifier.keras')

# 测试集路径
TEST_DIR = "../data/test"

# 数据预处理
datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # 确保图片顺序与真实标签一致
)

# 评估模型性能
loss, accuracy = model.evaluate(test_generator)
print(f"测试集损失: {loss:.4f}")
print(f"测试集准确率: {accuracy:.4f}")

# 获取预测结果
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)  # 将概率转换为 0 或 1
y_true = test_generator.classes  # 真实标签

# 找出错误分类的样本
errors = np.where(y_pred.flatten() != y_true)[0]
print(f"错误分类样本数: {len(errors)}")

# 保存错误分类结果为 CSV
error_data = [
    {"File": test_generator.filepaths[i],
     "True Label": "Dog" if y_true[i] else "Cat",
     "Predicted Label": "Dog" if y_pred[i] else "Cat"}
    for i in errors
]
df = pd.DataFrame(error_data)
df.to_csv("../results/error_analysis.csv", index=False)
print("错误分类结果已保存到 '../results/error_analysis.csv'")

# 可视化错误分类样本
def visualize_errors(test_generator, errors, num_errors_to_display=5):
    """
    可视化模型错误分类的样本
    :param test_generator: 测试数据生成器
    :param errors: 错误分类样本的索引
    :param num_errors_to_display: 展示的错误样本数量
    """
    for i, error_idx in enumerate(errors[:num_errors_to_display]):
        # 获取文件路径
        img_path = test_generator.filepaths[error_idx]
        true_label = "Dog" if y_true[error_idx] else "Cat"
        pred_label = "Dog" if y_pred[error_idx] else "Cat"

        # 加载并显示图片
        img = load_img(img_path)
        plt.imshow(img)
        plt.title(f"预测: {pred_label}, 真实: {true_label}", fontsize=12)
        plt.axis('off')
        plt.show()

# 可视化前 5 个错误分类样本
visualize_errors(test_generator, errors, num_errors_to_display=5)

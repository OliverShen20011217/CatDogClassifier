import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 加载模型
model = load_model('/Users/olivershen/PycharmProjects/CatDogClassifier/results/cat_dog_classifier.keras')

# 定义预测函数
def predict_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # 调整图片大小
    img_array = img_to_array(img) / 255.0  # 转换为数组并归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

# 测试预测
image_path = "/Users/olivershen/PycharmProjects/CatDogClassifier/data/test/cats/cat.4001.jpg"  # 替换为你的图片路径
result = predict_image(image_path)
print(f"预测结果: {result}")

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import csv
from datetime import datetime

# 加载模型
MODEL_PATH = './results/cat_dog_classifier.keras'
model = load_model(MODEL_PATH)

# 定义预测函数
def predict_image(image_path):
    """
    对单张图片进行预测
    :param image_path: 图片路径
    :return: 预测类别 ('Cat' or 'Dog') 和概率
    """
    img = load_img(image_path, target_size=(150, 150))  # 调整图片大小
    img_array = img_to_array(img) / 255.0  # 转换为数组并归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
    prediction = model.predict(img_array)
    prob = prediction[0][0]
    return ("Dog", prob) if prob > 0.5 else ("Cat", 1 - prob)

# 保存用户反馈的错误分类
def log_feedback(image_path, predicted_label, true_label):
    """
    保存用户反馈记录到 CSV 文件
    :param image_path: 图片路径
    :param predicted_label: 模型预测的类别
    :param true_label: 用户提供的真实类别
    """
    feedback_file = './feedback_log.csv'
    feedback_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 检查是否已经记录
    try:
        with open(feedback_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == image_path:
                    st.warning("此图片反馈已存在，无需重复提交。")
                    return
    except FileNotFoundError:
        pass

    # 记录反馈
    with open(feedback_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 如果文件不存在，写入标题行
        if file.tell() == 0:
            writer.writerow(["Image Path", "Predicted Label", "True Label", "Feedback Time"])
        writer.writerow([image_path, predicted_label, true_label, feedback_time])
        st.success(f"反馈已保存到文件: {feedback_file}")

# Streamlit 应用界面
st.title("🐾 Cat vs Dog Classifier 🐾")
st.write("上传一张图片，查看模型预测结果，并提供反馈！")

# 上传图片
uploaded_file = st.file_uploader("上传图片（支持 jpg, png 格式）", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 保存图片到本地
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 显示上传的图片
    st.image(image_path, caption="上传的图片", use_column_width=True)

    # 添加进度条
    with st.spinner("模型正在预测，请稍候..."):
        result, probability = predict_image(image_path)

    # 显示预测结果
    st.success("预测完成！")
    st.write(f"预测结果: **{result}**")
    st.write(f"预测概率: **{probability:.2%}**")

    # 提供反馈选项
    st.write("### 反馈：预测结果是否正确？")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 正确"):
            st.success("感谢您的反馈！")
    with col2:
        if st.button("❌ 错误"):
            true_label = st.radio("请选择真实类别:", ["Cat", "Dog"])  # 用户选择真实标签
            if true_label:
                log_feedback(image_path, result, true_label)

# 页面底部
st.markdown("---")
st.markdown("💡 **说明**: 这是一个基于深度学习的猫狗分类器，使用 Keras 和 TensorFlow 构建。")

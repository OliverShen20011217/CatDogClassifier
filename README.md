# Cat vs Dog Classifier 🐾
A deep learning-based image classifier to distinguish between cats and dogs using TensorFlow and Keras. This project demonstrates the complete pipeline of data preprocessing, model training, evaluation, and deployment through a Streamlit web application.

---

## 📌 Features
- Binary classification: Predicts whether an image contains a cat or a dog.
- Data preprocessing: Includes image resizing, normalization, and data augmentation.
- CNN architecture: Built with Conv2D layers, pooling layers, and dropout for regularization.
- Model evaluation: Provides accuracy and loss metrics for both training and validation datasets.
- Web interface: Deploys the trained model using Streamlit for user-friendly interaction.
- User feedback: Supports saving incorrect predictions for further model improvement.

---

## 📂 Project Structure
CatDogClassifier/ ├── data/ # Contains the dataset │ ├── train/ # Training and validation datasets │ ├── test/ # Test dataset ├── src/ # Source code │ ├── data_preprocessing.py │ ├── model_training.py │ ├── model_evaluation.py ├── notebooks/ # Optional Jupyter Notebooks ├── results/ # Saved model and logs │ ├── cat_dog_classifier.keras ├── feedback_log.csv # User feedback logs ├── app.py # Streamlit web application ├── README.md # Project documentation ├── requirements.txt # Python dependencies

yaml
复制
编辑

---

## 🛠️ Installation
Follow these steps to set up and run the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CatDogClassifier.git
   cd CatDogClassifier
Set up a virtual environment:

bash
复制
编辑
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies:

bash
复制
编辑
pip install -r requirements.txt
Run the Streamlit app:

bash
复制
编辑
streamlit run app.py
📊 Dataset
The dataset used in this project is from a publicly available source. The images are divided into two categories: cats and dogs.

Source: Kaggle: Microsoft Cats vs Dogs Dataset

The dataset is used for educational and non-commercial purposes.

🧠 Model Architecture
The convolutional neural network (CNN) model consists of:

Conv2D layers: For feature extraction.
MaxPooling2D layers: To reduce spatial dimensions.
Dropout layers: To prevent overfitting.
Dense layers: For classification.
Model summary:

scss
复制
编辑
Layer (type)                Output Shape              Param #   
=================================================================
Conv2D (32 filters, 3x3)    (None, 148, 148, 32)      896       
MaxPooling2D                (None, 74, 74, 32)        0         
Conv2D (64 filters, 3x3)    (None, 72, 72, 64)        18496     
MaxPooling2D                (None, 36, 36, 64)        0         
Conv2D (128 filters, 3x3)   (None, 34, 34, 128)       73856     
MaxPooling2D                (None, 17, 17, 128)       0         
Flatten                     (None, 36992)             0         
Dense (512 units)           (None, 512)               18940416  
Dropout                     (None, 512)               0         
Dense (1 unit)              (None, 1)                 513       
=================================================================
Total params: 19,034,177
📈 Results
Training Accuracy: 71.95%
Validation Accuracy: 72.44%
Test Accuracy: 74.39%
Example predictions:

Input:
→ Prediction: Dog
Input:
→ Prediction: Cat
🌐 Deployment
The project includes a Streamlit web app for real-time predictions:

Upload an image (jpg, png).
The app predicts whether it's a cat or a dog.
Option to provide feedback for incorrect predictions.
🤖 Feedback and Improvement
Users can mark incorrect predictions, and the app will save these feedback logs to feedback_log.csv.
Use these logs to retrain the model for better accuracy.
📋 Future Work
Add support for multi-class classification (e.g., other animal species).
Improve model accuracy with advanced architectures (e.g., ResNet or MobileNet).
Implement batch predictions for multiple images.
🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

📝 License
This project is for educational purposes only. All rights to the dataset belong to their respective owners.

🙋 Contact
For any questions or suggestions, please reach out:

Name: Oliver Shen
Email: shenzheyu1217@gmail.com
GitHub: https://github.com/OliverShen20011217
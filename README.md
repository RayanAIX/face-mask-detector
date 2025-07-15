# face-mask-detector
Deep Learning project using CNN to detect face masks in images
# 😷 Face Mask Detector – Deep Learning Project

This is a deep learning-based Face Mask Detector built using **Convolutional Neural Networks (CNN)** in TensorFlow/Keras.  
It classifies whether a person is wearing a mask or not using image data.

Trained on over **7,500 images** with GPU acceleration using **Google Colab**, this model achieved **93.9% validation accuracy**.

---

## 📂 Dataset

- ✅ `with_mask/` – 3,725 images  
- ✅ `without_mask/` – 3,828 images  
- Total: 7,553 images  
- Format: `.jpg` images with two class folders

---

## 🧠 Model Overview

- **Custom CNN** with:
  - 3 Convolutional Layers
  - Batch Normalization
  - Max Pooling
  - Dropout Regularization
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Accuracy Achieved:** 93.9% on validation set  
- **Model File:** `high_accuracy_mask_detector.h5`

---

## 📈 Features

- Data Augmentation with ImageDataGenerator
- Train/Test Split using Scikit-Learn
- EarlyStopping to prevent overfitting
- Normalized input data
- One-hot label encoding
- Visualized training/validation accuracy
- Exported trained model for deployment

---

## 🔍 Output Screenshot

> Final Training Result (93.9% Validation Accuracy)

[Training Output](https://github.com/RayanAIX/face-mask-detector/blob/2a3bb9c6e2f38d6f5eb0993d7a0c4c3644f39357/training_output.png)


---

## ⚙️ Libraries Used

- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 How to Use

1. Clone the repository  
2. Load the dataset with `with_mask/` and `without_mask/` folders  
3. Run the notebook: `face_mask_detector.ipynb` in Google Colab  
4. Trained model will be saved as `high_accuracy_mask_detector.h5`  
5. You can deploy it using:
   - Streamlit (web app)
   - OpenCV (real-time webcam detection)
   - Flask/Gradio (REST API or interface)

---

## 🌍 Future Work

- Real-time detection with webcam
- Deploy on Streamlit or Hugging Face Spaces
- Convert model to TFLite for mobile apps
- Integrate into security systems or thermal cameras

---

## ✨ Author

**Muhammad Rayan Shahid**  
Founder of [ByteBrilliance AI](https://www.youtube.com/@ByteBrillianceAI)  
💼 Future AI Engineer | 🇵🇰 Pakistan | 🧠 AI Enthusiast  


---

## 🏷️ Tags

`#DeepLearning` `#CNN` `#FaceMaskDetection` `#AI` `#TensorFlow` `#Keras` `#ComputerVision` `#ByteBrillianceAI`

---

## 🧠 Special Note

This project is part of my journey to become an **AI Engineer**, build a world-class **GitHub profile**, secure a **scholarship at MBZUAI**, and land a **remote AI job** at a top international company. 💥

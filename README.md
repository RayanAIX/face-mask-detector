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


![Training Output](training_output.png)

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


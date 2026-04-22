
# emotion-detection-project
# 🎭 Real-Time Emotion Detection using Deep Learning

## 📌 Overview

This project implements a **Real-Time Facial Emotion Recognition System** using **Convolutional Neural Networks (CNNs)** and **Computer Vision**.
It detects human facial expressions from webcam input and classifies them into multiple emotional categories.

---

## 🎯 Objectives

* Build a deep learning model to classify facial emotions
* Perform real-time emotion detection using webcam
* Improve model performance using data augmentation and optimization techniques

---

## 🧠 Features

* 🎥 Live emotion detection via webcam
* 🖼️ Image-based emotion prediction
* ⚙️ Data augmentation for better generalization
* 📊 Training & validation accuracy visualization
* 🧩 Modular and easy-to-understand code

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
* **Model Type:** Convolutional Neural Network (CNN)

---

## 📂 Project Structure

```id="1b5rsc"
emotion-detection-project/
│
├── train_model.py        # Model training script
├── emotion_live.py       # Real-time webcam detection
├── test_model.py         # Test on static image
├── test.jpg              # Sample image
├── requirements.txt      # Dependencies
├── README.md             # Documentation
│
├── fer2013/              # Dataset (not included)
│   ├── train/
│   └── test/
```

---

## 📥 Dataset

This project uses the **FER2013 dataset**.

👉 Download from:
https://www.kaggle.com/datasets/msambare/fer2013

After downloading, place it as:

```id="b4nx0r"
fer2013/
   ├── train/
   └── test/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```id="b0hv3o"
git clone https://github.com/ranjani-hegde/emotion-detection-project.git
cd emotion-detection-project
```

### 2️⃣ Install dependencies

```id="o1mw7s"
pip install -r requirements.txt
```

---

## 🧠 Model Training

```id="9ev01m"
python train_model.py
```

👉 This will train the CNN model and save it locally.

---

## 🖼️ Test on Image

```id="j6x6yy"
python test_model.py
```

---

## 🎥 Real-Time Emotion Detection

```id="c7nh5t"
python emotion_live.py
```

👉 Press **Q** to exit webcam

---

## 📊 Model Performance

* Training Accuracy: ~50–55%
* Validation Accuracy: ~50–56%

*(Performance can be improved with advanced techniques)*

---

## 🚀 Future Enhancements

* Improve accuracy using Transfer Learning (VGG16, ResNet)
* Build GUI using Streamlit or Tkinter
* Deploy as a web application
* Optimize model for real-time performance

---

## ⚠️ Notes

* Dataset and trained model are not included due to size constraints
* Ensure webcam access is enabled
* Use `compile=False` while loading `.h5` model if needed

---

## 👩‍💻 Author

**Ranjani Hegde**

---

## ⭐ Acknowledgements

* FER2013 Dataset
* TensorFlow & OpenCV communities

---

## 📌 If you found this useful

Give this project a ⭐ on GitHub!


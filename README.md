
# 🧠 AI vs Real Image Detector

An intelligent web-based application that detects whether an image is **AI-generated 🤖** or **real 📸** using deep learning and image analysis techniques.

---

## 🚀 Demo Preview

<img width="1718" height="912" alt="image" src="https://github.com/user-attachments/assets/bd70f28c-494b-4e98-b4b2-3afb09e24b00" />
---
<img width="1592" height="335" alt="image" src="https://github.com/user-attachments/assets/0de14287-df40-4860-a7b0-e5e712d8c620" />

> Upload an image and instantly get prediction with confidence %

---

## 🔍 Features

- ✅ Detects AI-generated vs Real images  
- 📊 Provides **confidence percentage**  
- 🧠 Uses **deep learning (ResNet18 CNN model)**  
- 🧾 Analyzes **image metadata (EXIF data)**  
- 🌫️ Detects **noise patterns** in images  
- ⚡ Fast and lightweight processing  
- 🌐 Built with FastAPI backend  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- PyTorch  
- OpenCV  
- NumPy  
- Pillow  
- EXIFRead  

---

## ⚙️ How It Works

This project uses a **multi-layer detection approach**:

1. **Deep Learning Model (ResNet18)**  
   - Extracts features from the image  
   - Detects AI patterns  

2. **Metadata Analysis**  
   - Checks EXIF data  
   - AI images often lack metadata  

3. **Noise Analysis**  
   - Real images have natural noise  
   - AI images are often smoother  

4. **Final Prediction**  
   - Combines all scores  
   - Outputs result with confidence %

---

## 📂 Project Structure


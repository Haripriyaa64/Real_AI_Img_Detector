
# 🧠 AI vs Real Image Detector

An intelligent web-based application that detects whether an image is **AI-generated 🤖** or **real 📸** using deep learning and image analysis techniques.


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




# 🧠 AI Assistant for Radiology Images

An AI-powered web application for analyzing **chest X-ray images** using deep learning. The system predicts multiple possible pathologies and provides **Grad-CAM visual explanations** to highlight regions influencing the model’s decisions.

---

## 🚀 Live Demo

👉 https://chest-xray-ai-assistant-jhp2fmx8zatbvojkemcvkg.streamlit.app/

---

## 📌 Overview

This application uses a pretrained **DenseNet model** from `torchxrayvision` to perform **multi-label classification** on chest X-rays.

It enables users to upload medical images and receive:

* Disease probability predictions
* Visual explanations using Grad-CAM

---

## ✨ Features

* 🖼️ Upload **JPG, PNG, or DICOM (.dcm)** chest X-rays
* ⚙️ Automatic preprocessing (including DICOM windowing)
* 🤖 Multi-label prediction of chest pathologies
* 🔥 Grad-CAM heatmap visualization
* 📊 Probability-based results display
* 🌐 Interactive UI built with Streamlit

---

## 🛠️ Tech Stack

* Python
* PyTorch
* TorchXRayVision
* OpenCV
* Streamlit

---

## ⚙️ How It Works

1. User uploads an X-ray image
2. Image is preprocessed and normalized
3. Passed through DenseNet model
4. Model outputs probabilities for multiple diseases
5. Grad-CAM generates heatmap for interpretability
6. Results displayed in UI

---

## 📦 Installation (Local Setup)

```bash
git clone https://github.com/Ishant713/Chest-xray-ai-assistant.git
cd Chest-xray-ai-assistant
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
app.py              # Streamlit interface
inference.py        # Model loading & prediction logic
utils.py            # Image preprocessing utilities
requirements.txt    # Python dependencies
packages.txt        # System dependencies
runtime.txt         # Python version configuration
```

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**.
It is **NOT a medical device** and must not be used for diagnosis or treatment.

---

## 📌 Notes

* Supports **posterior-anterior (PA/AP) chest X-rays**
* Not designed for CT, MRI, or ultrasound
* Model weights are downloaded automatically on first run
* No personal health information (PHI) should be used

---

## 🎯 Future Improvements

* 📄 Generate downloadable medical reports
* 📊 Add evaluation metrics (ROC, accuracy)
* 🧠 Fine-tune model on custom datasets
* 🔐 Add user authentication system

---

## 👨‍💻 Author

**Ishan Dhakad**
Aspiring AI Engineer | Data Science Enthusiast

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

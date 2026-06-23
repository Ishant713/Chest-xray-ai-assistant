# 🧠Chest X-Ray AI Assistant

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
## Project Demo
### Example 1 
<img width="1615" height="787" alt="Screenshot 2026-06-24 003618" src="https://github.com/user-attachments/assets/ef015cb8-c316-42e5-a634-5ffb69c3cccc" />
<img width="1795" height="669" alt="Screenshot 2026-06-24 003736" src="https://github.com/user-attachments/assets/2479f79a-182e-433c-a766-244c927fa4b4" />
<img width="1685" height="775" alt="Screenshot 2026-06-24 003827" src="https://github.com/user-attachments/assets/141f4aa4-4a1b-449a-a62c-7cf60d74194a" />

### Example 2
<img width="1766" height="765" alt="Screenshot 2026-06-24 005607" src="https://github.com/user-attachments/assets/f8afed6d-a6d0-4800-ae39-b1e0fd9ea7a8" />
<img width="1814" height="678" alt="Screenshot 2026-06-24 005620" src="https://github.com/user-attachments/assets/633abd6a-e749-4d41-9f63-fc3485f63c01" />
<img width="1755" height="762" alt="Screenshot 2026-06-24 005630" src="https://github.com/user-attachments/assets/05def0ff-516e-401d-9c7c-86ac9d92d83b" />





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

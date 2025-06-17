# ANOMALY-DETECTION-USING-STRUCTURAL-SIMILARITY-METRICS-AND-YOLO-V8-

# 🔍 NCLR - Neural Classification with Lightweight Runtime

This project provides an integrated workflow for training and deploying a neural network classifier using Streamlit for the frontend and Google Colab for backend training. The model is designed for efficient inference using pre-trained weights.

---

## 📁 Project Structure

├── nclr.py # Streamlit app - loads model and provides UI
├── lrnc.ipynb # Google Colab notebook - model training backend
├── weights/
│ ├── best.pt # Best performing trained model
│ └── least.pt # Least performing model for reference



---

## ⚙️ Features

- 🔍 **YOLOv8-based object detection**
- 🧠 Training in **Google Colab** using `Ultralytics` library
- 📦 Pretrained model weights (`.pt`) stored in the `weights/` folder
- 🖥️ **Streamlit app** for interactive predictions

---

## 🚀 How to Use

### 1. 🧪 Train the Model in Google Colab
- Open `lrnc.ipynb` in [Google Colab](https://colab.research.google.com/).
- Train a custom YOLOv8 model using Ultralytics:
  ```python
  !pip install ultralytics
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')  # Or yolov8s/m/l/x.pt
  model.train(data='data.yaml', epochs=50)




---

## 🚀 How to Use

### 1. 🔧 Training the Model (Google Colab)
- Open the `lrnc.ipynb` notebook in **Google Colab**.
- Train your model or modify the training parameters as needed.
- After training, export the model as `.pt` files.
- Store the `best.pt` and `least.pt` inside the `weights/` folder.

### 2. 🖥️ Running the Streamlit App
- Ensure `nclr.py` and the `weights/` folder are in the same directory.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
> -->   streamlit run nclr.py


🧠 Model Details
The model is trained using PyTorch and saved in .pt format.

Inference is handled inside the nclr.py Streamlit interface.

Streamlit provides a simple and interactive way to test your model with new inputs.


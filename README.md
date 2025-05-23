# 🌊 Flood Risk Prediction Web App

A web application for predicting flood risk using both classification and regression machine learning models. Built with FastAPI, styled with HTML/CSS.

## 📦 Features

- 🔍 **Classification**: Predicts flood risk as binary output (0 or 1).
- 📈 **Regression**: Predicts a continuous flood risk score.
- 🖥️ Clean and user-friendly web interface for data input and result display.
- 🧠 Uses pretrained models: `flood_model.pkl` (classifier) and `flood_regressor.pkl` (regressor).
- 📊 Inputs are scaled using `flood_scaler.pkl`.


## ⚙️ Technologies Used

- Python 3.11
- FastAPI
- scikit-learn
- Jinja2 (templating)
- HTML + CSS
- Git LFS (for model storage)

## 🧪 Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/flood-risk-app.git

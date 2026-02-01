# 📡 Optical Systems ML Dashboard

An **Explainable Machine Learning Dashboard** built with **Streamlit** to model and predict the behavior of **Optical Communication Systems**.

This project uses Machine Learning to predict:

* 📈 **BER (Bit Error Rate)** — using Regression
* 🩺 **System Health Status** — using Classification
* 🔍 **Model Explainability** — using SHAP (SHapley Additive Explanations)

The app allows users to interactively change optical parameters and instantly observe how system performance is affected.

---

## 🚀 Live Demo

👉 **Deployed App Link: https://jovac-project-zg37dkycye6gzfr4op62zq.streamlit.app/

---

## 🧠 Problem Statement

In optical communication systems, parameters such as:

* Input Power
* Dispersion
* Refractive Index
* Curvature
* Wavelength

directly affect signal quality and system reliability.

Traditionally, analyzing their combined effect requires complex physical modeling.
This project replaces that with a **data-driven ML approach** that is:

* Fast
* Accurate
* Explainable

---

## 🧪 What This Project Does

1. Accepts optical system parameters via an interactive UI
2. Computes derived physical features:

   * Attenuation (dB)
   * Focal Length
3. Uses trained ML models to:

   * Predict **Downstream BER**
   * Predict **System Health Condition**
4. Uses **SHAP** to explain which parameters most influenced the prediction

---

## 🖥️ Tech Stack

| Technology     | Purpose                   |
| -------------- | ------------------------- |
| Python         | Core language             |
| Streamlit      | Interactive dashboard     |
| scikit-learn   | ML pipelines              |
| XGBoost        | High-performance modeling |
| SHAP           | Explainable AI            |
| Pandas / NumPy | Data processing           |
| Matplotlib     | Visualization             |
| Joblib         | Model serialization       |

---

## 📂 Project Structure

```
jovac-project/
│
├── app.py                         # Streamlit dashboard
├── ML_P.ipynb                     # Model training & experimentation
├── best_regression_model.pkl      # BER prediction model
├── best_classification_model.pkl  # System health model
├── label_encoder.pkl              # Encoder for classification labels
├── requirements.txt               # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/srthk-13/jovac-project.git
cd jovac-project
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🎛️ Input Parameters in the Dashboard

Users can control:

* Input Power (mW)
* Dispersion (ps/nm)
* Refractive Index
* Curvature (mm)
* Wavelength (µm)

The app automatically computes:

* Attenuation (dB)
* Focal Length

---

## 📊 Predictions Provided

### 📈 BER Prediction

Predicts Bit Error Rate of the optical signal using a regression model.

### 🩺 System Health Classification

Classifies the optical system into health categories using a classification model.

---

## 🔍 Explainable AI with SHAP

The dashboard provides SHAP summary plots to explain:

* Which optical parameters influence BER the most
* Which parameters affect system health

This makes the ML model **transparent and trustworthy**.

---

## 🧠 Model Training

All models were trained and evaluated in `ML_P.ipynb` using various algorithms.
The best performing models were saved and integrated into the app.

---

## 🌟 Key Highlights

✅ Applied ML in Optical / Photonics domain
✅ Real-time interactive predictions
✅ Derived physics-based features
✅ Explainable AI integration (SHAP)
✅ Clean Streamlit dashboard

---

## 📌 Future Improvements

* Add real dataset upload support
* Add more optical parameters
* Deploy with Docker
* Improve SHAP performance for large samples

---

## 👤 Author

**Sarthak**
GitHub: [https://github.com/srthk-13](https://github.com/srthk-13)

---

## 📄 License

This project is open-source and available under the MIT License.

# 🎓 Teaching Assistant Performance Prediction — PRCP-1026

A machine learning web application that predicts the performance category (Low, Medium, High) of Teaching Assistants using course, instructor, and class-related features.

---

## 🚀 Live Demo

👉 https://your-render-link.onrender.com

---

## 📊 Project Overview

| Feature        | Details                                  |
|----------------|------------------------------------------|
| Project ID     | PRCP-1026                                |
| Type           | Multi-Class Classification               |
| Dataset        | Teaching Assistant Evaluation Dataset    |
| Target         | Class (Low, Medium, High)                |
| Best Model     | Random Forest                            |
| Metric         | Weighted F1 Score                        |

---

## 🏆 Model Results

| Model               | Performance |
|--------------------|------------|
| Random Forest      | ⭐ Best     |
| XGBoost            | Strong     |
| LightGBM           | Strong     |
| Gradient Boosting  | Good       |
| Logistic Regression| Baseline   |

---

## 📁 Project Structure
teaching-assistance-ml-project/
├── Data/tae.csv
├── models/
│ ├── ta_best_model.pkl
│ └── ta_scaler.pkl
├── PRCP-1026-TeachingAssistance.ipynb
├── app.py
├── requirements.txt
└── README.md


---

## 🛠️ Run Locally
git clone https://github.com/uwaiszmuhammed07-hash/teaching-assistance-ml-project.git

cd teaching-assistance-ml-project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py


---

## 🎯 Key Highlights

- End-to-End ML Pipeline  
- Feature Engineering  
- Model Comparison  
- Streamlit Web App  
- Deployed on Render  

---

## 👤 Author

Uwais Muhammed KP  
Data Science Capstone Project — PRCP-1026  


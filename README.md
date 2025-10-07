# 🧠 Disease Predictor AI  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RahulBansal-24/Disease-Predictor-AI/blob/main/Disease_Predictor_AI.ipynb)

---

## 💖 Overview

**Disease Predictor AI** is a machine learning project that predicts whether a patient is likely to have **heart disease** based on health indicators such as age, cholesterol level, resting blood pressure, and more.  
The project experiments with multiple ML algorithms — **Logistic Regression** and **Random Forest Classifier** — to determine which model performs best for accurate prediction.  
After comparison, **Random Forest** was selected as the final deployed model.

Built and trained entirely in **Google Colab**, the project demonstrates a complete **end-to-end ML workflow**, including preprocessing, training, evaluation, and prediction.

---

## 🧩 Key Features

- 🩺 **Heart Disease Prediction** – Predicts whether a patient has heart disease based on medical data  
- ⚙️ **Multiple ML Models** – Implements and compares Logistic Regression and Random Forest  
- 📊 **Data Visualization** – Visualizes correlations, feature distributions, and feature importance  
- 💾 **Reusable Artifacts** – Includes trained Random Forest model, scaler, and sample CSV files  
- ☁️ **Google Colab Ready** – Can be run directly in the browser without local setup  

---

## 📁 Repository Contents

| File Name | Description |
|------------|-------------|
| `Disease_Predictor_AI.ipynb` | Main Google Colab notebook containing preprocessing, training (Logistic Regression + Random Forest), evaluation, and prediction |
| `heart_disease_rf_model.pkl` | Final trained Random Forest model used for prediction |
| `heart_disease_scaler.pkl` | StandardScaler object for consistent input feature scaling |
| `heart_disease_uci.csv` | Primary dataset used for model training (UCI Heart Disease dataset) |
| `heart_dataset_sample.csv` | Small sample dataset for quick testing or demonstration |
| `User_template.csv` | Blank input template for users to add their own test data |
| `LICENSE` | GNU General Public License (GPL) for open-source usage |

---

## 🧠 How It Works

1. **Data Preprocessing**  
   - Handles missing values and categorical encoding  
   - Scales features using `StandardScaler`  
   - Splits data into training and testing sets  

2. **Model Training & Comparison**  
   - Trains both **Logistic Regression** and **Random Forest Classifier**  
   - Compares performance using accuracy, precision, recall, and confusion matrix  
   - **Random Forest** achieves the highest accuracy and is saved as the final model  

3. **Prediction Phase**  
   - Loads the trained Random Forest model (`heart_disease_rf_model.pkl`)  
   - Applies scaling using the saved scaler  
   - Predicts whether a patient has heart disease based on new input data  

---

## 🚀 Run the Project in Google Colab

You can open and run the notebook directly in Colab using the badge above, or follow these steps:

1. Open [Google Colab](https://colab.research.google.com/)  
2. Go to **File → Open Notebook → GitHub tab**  
3. Paste URl: https://github.com/RahulBansal-24/Disease-Predictor-AI
4. Open `Disease_Predictor_AI.ipynb` and run all cells  

---

## 📊 Example Workflow

1. Load the dataset (`heart_disease_uci.csv`)  
2. Train and evaluate both models in the notebook  
3. Save the Random Forest model and scaler using `joblib`  
4. Use `User_template.csv` to format user input  
5. Predict on new data using the trained model  

---

## ⚠️ Important Notes

- The **Kaggle API key is not required** — the dataset is already included in the repository.  
- The model is built for **educational and research purposes only**.  
- Do **not** use this model for actual medical or diagnostic decisions.  

---

## 🧰 Technologies Used

- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Google Colab  

---

## 🔮 Future Improvements

- Add hyperparameter tuning for both models  
- Integrate with a web UI (Streamlit or Flask) for real-time prediction  
- Test additional algorithms like XGBoost or Support Vector Machines  
- Implement model explainability with SHAP or LIME  

---

## 🧑‍💻 Author

**Rahul Bansal**  
💼 GitHub: [@RahulBansal-24](https://github.com/RahulBansal-24)  
📧 For feedback or collaboration, feel free to connect!  

---

## 📜 License

This project is licensed under the **GNU General Public License (GPL)** — see the [LICENSE](LICENSE) file for full details.

---

### ❤️ Support

If you found this project useful, please ⭐ **star the repository** to show your support and help others discover it!


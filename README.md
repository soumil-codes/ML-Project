# 🎓 Student Math Score Predictor

An end-to-end Machine Learning project that predicts a student's **math score** based on demographic information, parental education level, lunch type, test preparation course status, and their reading and writing scores. The project covers the complete ML lifecycle — from data ingestion and preprocessing to model training, evaluation, and deployment as a web application.

---

## 🚀 Live Demo

🔗 **[Live Application](https://studentscorepredictor-mnf9.onrender.com/)**

---

## ✨ Key Features

- Modular, production-style ML pipeline (ingestion → transformation → training → prediction)
- Automated preprocessing using `Pipeline` and `ColumnTransformer`
- Multiple regression models trained and evaluated with automatic best-model selection
- Hyperparameter tuning for improved model performance
- Reusable prediction pipeline powered by serialized model and preprocessor artifacts
- Flask-based web interface for real-time predictions
- Custom exception handling and centralized logging
- Separate dependency management for production and development environments
- Deployed on Render using Gunicorn as the WSGI server

---

## 🛠️ Tech Stack

| Category            | Technologies                                   |
|----------------------|------------------------------------------------|
| Language             | Python                                          |
| Data Handling        | NumPy, Pandas                                   |
| Machine Learning      | Scikit-learn, XGBoost, CatBoost                 |
| Visualization         | Matplotlib, Seaborn                             |
| Experimentation       | Jupyter Notebook                                |
| Web Framework          | Flask                                          |
| Frontend               | HTML, CSS                                      |
| Model Serialization   | Pickle, Dill                                    |
| Deployment Server     | Gunicorn                                       |
| Hosting               | Render                                          |
| Version Control        | Git, GitHub                                    |

---

## 📊 Dataset and Features

The model is trained to predict a student's **math score** using the following input features:

| Feature                     | Type          | Description                                  |
|------------------------------|---------------|-----------------------------------------------|
| Gender                        | Categorical   | Student's gender                             |
| Race/Ethnicity                | Categorical   | Student's ethnic group                       |
| Parental Level of Education   | Categorical   | Highest education level attained by parent   |
| Lunch                          | Categorical   | Type of lunch (standard / free-reduced)      |
| Test Preparation Course        | Categorical   | Whether the test prep course was completed   |
| Reading Score                  | Numerical     | Student's reading score                      |
| Writing Score                  | Numerical     | Student's writing score                      |

**Target Variable:** `Math Score`

---

## 🔄 Machine Learning Workflow

```
Raw Data
   │
   ▼
Data Ingestion (train/test split)
   │
   ▼
Data Transformation
 (ColumnTransformer + Pipeline)
   │
   ▼
Model Training
 (Multiple Regression Models)
   │
   ▼
Model Evaluation (R² Score)
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Best Model Selection
   │
   ▼
Save Artifacts
 (model.pkl, preprocessor.pkl)
   │
   ▼
Prediction Pipeline
   │
   ▼
Flask Web Application
```

---

## 🧹 Data Preprocessing Details

Preprocessing is handled using Scikit-learn's `Pipeline` and `ColumnTransformer` to ensure consistent transformations between training and inference.

**Numerical Features** (`reading_score`, `writing_score`):
- Missing value imputation
- Feature scaling using `StandardScaler`

**Categorical Features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`):
- Missing value imputation
- Encoding using `OneHotEncoder`
- Scaling applied post-encoding

The fitted preprocessing pipeline is serialized and stored as `preprocessor.pkl` for reuse during inference.

---

## 🤖 Models Used

Multiple regression algorithms are trained and evaluated to identify the best-performing model, including models from Scikit-learn, along with **XGBoost** and **CatBoost** regressors. Model performance is compared using the **R² score**, and the best-performing model is automatically selected and persisted.

---

## 🎯 Hyperparameter Tuning

Each candidate model undergoes hyperparameter tuning to optimize its performance before the final comparison and selection step. The best combination of hyperparameters is chosen based on evaluation results on the test set.

---

## 🔮 Prediction Pipeline

The `predict_pipeline.py` module handles inference by:

1. Loading the saved `preprocessor.pkl` and `model.pkl` from the `artifacts/` directory
2. Transforming raw user input using the saved preprocessing pipeline
3. Feeding the transformed data into the trained model
4. Returning the predicted math score

This ensures that the exact transformations applied during training are consistently applied during prediction.

---

## 🌐 Web Application

A Flask web application (`application.py`) provides a simple interface for users to input student details and receive a predicted math score in real time.

- `templates/index.html` — Landing page
- `templates/home.html` — Prediction form and result display page

---

## 📁 Project Structure

```
ML-Project/
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
├── notebooks/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── application.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.x installed
- Git installed

### Clone the Repository

```bash
git clone https://github.com/soumil-codes/ML-Project.git
cd ML-Project
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Dependencies

For running the application:
```bash
pip install -r requirements.txt
```

For development/notebook work:
```bash
pip install -r requirements-dev.txt
```

---

## ▶️ How to Run Locally

1. Ensure the trained artifacts (`model.pkl`, `preprocessor.pkl`) exist in the `artifacts/` directory. If not, run the training pipeline first via the components in `src/components/`.
2. Start the Flask application:

```bash
python application.py
```

3. Open your browser and navigate to:

```
http://127.0.0.1:5000
```

4. Fill in the student details on the form and get the predicted math score.

---

## ☁️ Deployment

This project is deployed on **Render** using **Gunicorn** as the production WSGI server.

**Start command used on Render:**

```bash
gunicorn application:app
```

Steps to deploy:
1. Push the project to a GitHub repository
2. Create a new **Web Service** on Render and connect your GitHub repository
3. Set the build command to install dependencies from `requirements.txt`
4. Set the start command to `gunicorn application:app`
5. Deploy and access the app via the generated Render URL

---

## 📦 Dependency Management

Dependencies are split into two files to keep the production environment lightweight:

| File                    | Purpose                                                |
|--------------------------|---------------------------------------------------------|
| `requirements.txt`        | Core dependencies required to run the Flask application |
| `requirements-dev.txt`    | Additional dependencies for Jupyter Notebook / development and experimentation |

---

## 🧾 Logging and Exception Handling

- **`src/logger.py`** — Configures centralized logging to track pipeline execution, making it easier to debug and monitor each stage of the workflow.
- **`src/exception.py`** — Implements a custom exception class that captures detailed error information (file name, line number, and error message) for clearer debugging across the ingestion, transformation, and training stages.

---

## 🔭 Future Improvements

- Add CI/CD pipeline for automated testing and deployment
- Add unit tests for pipeline components
- Containerize the application using Docker
- Add model monitoring and retraining workflow
- Improve frontend UI/UX

---

## 👤 Author

**Soumil Jindal**
📧 soumiljindal2238@gmail.com
🔗 [GitHub](https://github.com/soumil-codes)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it with attribution.

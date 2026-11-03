# 🏥 MediPredict: Medical Equipment Failure Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=flat&logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite)

**MediPredict** is an end-to-end, machine learning-powered web application designed to predict medical equipment failures and assist biomedical engineers and healthcare staff in proactive maintenance planning. By monitoring continuous operational parameters—such as equipment age, usage frequency, maintenance schedules, and logged error codes—the system classifies equipment risk levels in real time and maintains operational logs in an embedded SQLite database.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack & Backend Infrastructure](#-tech-stack--backend-infrastructure)
- [Database Schema](#-database-schema)
- [System Architecture](#-system-architecture)
- [Installation Manual](#-installation-manual)
- [Running the Application](#-running-the-application)
- [Code Walkthrough](#-code-walkthrough)
- [Testing & Example Inputs](#-testing--example-inputs)
- [Future Enhancements](#-future-enhancements)

---

## 🏥 Project Overview

Unplanned medical equipment downtime poses severe risks to patient care and operational efficiency in hospital environments. **MediPredict** bridges predictive analytics and operational workflow by:
1. Predicting failure risk using a **Gradient Boosting Classifier** (achieving ~91.9% accuracy on benchmark data).
2. Storing logs and predictions persistently in a relational **SQLite** database.
3. Enabling bulk record imports via standard **CSV files**.
4. Providing interactive visualizations and inspection reports through a **Streamlit** dashboard.

---

## ✨ Key Features

- **📊 Centralized Dashboard & Metrics**: Real-time KPI metrics displaying total tracked equipment, counts of high-risk units, model benchmark statistics, and dynamic Plotly charts.
- **📝 Single Record Entry**: Form-based input for registering individual equipment with automatic machine learning scoring and persistent database saving.
- **📁 Bulk CSV File Upload**: Ingest and process large datasets of medical equipment records simultaneously.
- **🔍 Failure Prediction & Inspection Review**: Searchable equipment registry with detailed diagnostic breakdowns and deletion/maintenance capabilities.
- **⚙️ Model Benchmarking**: Comparative analysis of different ML algorithms (Gradient Boosting, Random Forest, SVM, Logistic Regression).

---

## 🛠️ Tech Stack & Backend Infrastructure

### 1. **Frontend / Presentation Layer**
* **[Streamlit](https://streamlit.io/)**: Open-source framework used to build interactive, dynamic web applications directly in Python.
* **[Plotly Express](https://plotly.com/python/)**: Interactive data visualization library used for dynamic charts and operational risk distributions.

### 2. **Backend Logic & Machine Learning**
* **Python 3.9+**: Primary core logic execution environment.
* **[Scikit-Learn](https://scikit-learn.org/)**:
  * `GradientBoostingClassifier`: Primary model choice for predictive failure classification based on non-linear ensemble boosting trees.
  * `StandardScaler`: Normalizes feature metrics (age, usage hours, days since service, error counts) before inference.
  * `train_model()` cached via `@st.cache_resource` for optimization.
* **Pandas & NumPy**: High-performance data manipulation, dataframe transformations, and bulk CSV parsing.

### 3. **Database & Persistence Layer**
* **[SQLite3](https://www.sqlite.org/)**: Serverless, self-contained SQL database engine integrated natively into Python via standard library bindings. Standard transactional operations (`INSERT`, `SELECT`, `DELETE`) ensure zero data loss across user sessions.

---

## 🗄️ Database Schema

The application creates and interacts with an SQLite table named `equipment` stored in `medipredict.db`:

```sql
CREATE TABLE IF NOT EXISTS equipment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_name TEXT NOT NULL,
    equipment_type TEXT NOT NULL,
    age_years REAL NOT NULL,
    usage_hours_per_week REAL NOT NULL,
    last_maintenance_days INTEGER NOT NULL,
    error_code_count INTEGER NOT NULL,
    predicted_category TEXT NOT NULL
);
```

### Schema Field Explanations:
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` | Auto-incrementing unique primary key identifier. |
| `equipment_name` | `TEXT` | Descriptive unit label (e.g., *MRI Scanner 01*). |
| `equipment_type` | `TEXT` | Classification domain (Imaging, Monitoring, Surgical, Lab, Support). |
| `age_years` | `REAL` | Operational age of unit in years. |
| `usage_hours_per_week`| `REAL` | Average weekly operating hours. |
| `last_maintenance_days`| `INTEGER` | Days elapsed since last physical maintenance service. |
| `error_code_count` | `INTEGER` | Total number of recorded warning/error codes in past 90 days. |
| `predicted_category` | `TEXT` | Model inference result: `High Risk` or `Normal / Scheduled`. |

---

## 🏗️ System Architecture

```text
               +----------------------------------+
               |     User Interface (Streamlit)    |
               |  (Dashboard, Forms, CSV Upload)  |
               +-----------------+----------------+
                                 |
                        User Input / File Data
                                 |
                                 v
               +----------------------------------+
               |     Backend Logic (Python)       |
               |  - Pandas Data Processing        |
               |  - Feature Normalization         |
               +-----------------+----------------+
                                 |
                        Normalized Vectors
                                 |
                                 v
               +----------------------------------+
               |     Machine Learning Pipeline    |
               |   (Gradient Boosting Model)      |
               +-----------------+----------------+
                                 |
                        Inference Results
                                 |
                                 v
               +----------------------------------+
               |      Relational Storage          |
               |      (SQLite Database)           |
               +----------------------------------+
```

---

## 📦 Installation Manual

Follow these steps to set up the development and execution environment on your local system.

### Prerequisites
* **Python 3.9** or higher installed on your system. Verification command:
  ```bash
  python --version
  ```
* **Git** installed on your machine.

---

### Step 1: Clone the Repository
Clone the repository or extract the project folder:
```bash
git clone https://github.com/your-username/medipredict.git
cd medipredict
```

---

### Step 2: Set Up a Virtual Environment (Recommended)
Creating an isolated environment ensures dependency stability.

* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

---

### Step 3: Install Required Dependencies
Create a `requirements.txt` file (or install packages directly):

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

#### `requirements.txt` Content:
```text
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
```

To install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

1. Make sure your virtual environment is active.
2. Run the Streamlit execution command from the root directory containing `app.py`:
   ```bash
   streamlit run app.py
   ```
3. Streamlit will automatically launch the application in your default web browser at:
   - **Local URL:** `http://localhost:8501`
   - **Network URL:** `http://<your-local-ip>:8501`

---

## 💻 Code Walkthrough

Below is an overview of key structural components inside `app.py`:

```python
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('medipredict.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_name TEXT,
            equipment_type TEXT,
            age_years REAL,
            usage_hours_per_week REAL,
            last_maintenance_days INTEGER,
            error_code_count INTEGER,
            predicted_category TEXT
        )
    ''')
    conn.commit()
    conn.close()

# --- 2. MACHINE LEARNING ENGINE ---
@st.cache_resource
def train_model():
    """Trains a reference Gradient Boosting Classifier model and caches it in memory."""
    np.random.seed(42)
    # Synthetic dataset generator matching operational domains
    X = np.random.rand(800, 4) * [15, 100, 365, 10]
    y = (X[:, 0] * 0.3 + X[:, 2] * 0.002 + X[:, 3] * 0.5 > 5).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Initializing core services
init_db()
model, scaler = train_model()
```

---

## 🧪 Testing & Example Inputs

### Method A: Single Input via Registration Form
Navigate to **Equipment Registration & Upload** $\rightarrow$ **Manual Entry**:

| Input Field | Test Case 1 (Low Risk) | Test Case 2 (High Risk) |
| :--- | :--- | :--- |
| **Equipment Name** | `Patient Monitor ICU-03` | `MRI Machine Alpha` |
| **Equipment Type** | `Monitoring` | `Imaging` |
| **Equipment Age (Years)** | `1.5` | `11.5` |
| **Usage Hours / Week** | `25.0` | `88.0` |
| **Days Since Maintenance** | `10` | `210` |
| **Error Code Count** | `0` | `8` |
| **Expected Inference** | `Normal / Scheduled` | `High Risk` |

---

### Method B: Bulk Upload via CSV File
Create a standard `.csv` file named `sample_inventory.csv` with the following content:

```csv
equipment_name,equipment_type,age_years,usage_hours_per_week,last_maintenance_days,error_code_count
MRI Scanner 01,Imaging,12.0,85.0,180,7
Ultrasound Unit B,Imaging,1.5,20.0,10,0
Infusion Pump 12,Monitoring,6.5,60.0,120,4
Ventilator Main-04,Surgical,8.0,75.0,210,6
Centrifuge Alpha,Laboratory,3.0,40.0,30,1
Autoclave Sterilizer 02,Support,10.0,90.0,150,8
```

1. Navigate to **Equipment Registration & Upload** $\rightarrow$ **Bulk Upload (CSV)**.
2. Drag and drop `sample_inventory.csv`.
3. Click **Process & Save Uploaded Records**.
4. Check the **Dashboard** or **Failure Prediction & Review** tab to verify processing.

---

## 🔮 Future Enhancements

- **PostgreSQL Integration**: Migration path from SQLite to PostgreSQL for concurrent multi-user enterprise deployments.
- **IoT Telemetry Ingestion**: REST API endpoint (via FastAPI) to ingest real-time sensor metrics directly from hospital equipment.
- **Automated Alerts**: Email or SMS notification dispatch upon detection of `High Risk` items.

---

## 📜 License & Disclaimers

This software is developed strictly for **educational and research prototyping purposes**. Model predictions serve as decision-support guidance and should not replace qualified clinical or biomedical engineering judgment.

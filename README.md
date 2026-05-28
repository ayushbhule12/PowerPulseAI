PowerPulse AI
Smart Household Electricity Consumption Forecasting and Anomaly Detection Using Machine Learning & Deep Learning
Project Overview

PowerPulse AI is an end-to-end time-series analytics project designed to forecast household electricity consumption and detect abnormal energy usage patterns using Machine Learning, Deep Learning, and Statistical Time-Series techniques.

The project uses the UCI Individual Household Electric Power Consumption Dataset and builds a complete intelligent analytics pipeline that includes:

Data preprocessing
Exploratory Data Analysis (EDA)
Time-series feature engineering
Forecasting models
Deep learning models
Anomaly detection
Model evaluation
Interactive Streamlit dashboard

This project demonstrates practical applications of:

Time-series forecasting
Energy analytics
AI-powered anomaly detection
Smart monitoring systems
Deep learning for sequential data
Features
Data Processing
Missing value handling
Datetime parsing
Hourly resampling
Feature cleaning
Exploratory Data Analysis
Consumption trends
Monthly and hourly patterns
Correlation heatmaps
Statistical summaries
Feature Engineering
Time-based features
Lag features
Rolling statistics
Cyclical encoding
Forecasting Models
Random Forest Regressor
XGBoost Regressor
SARIMA
LSTM
GRU
Anomaly Detection
Z-Score Detection
Rolling Threshold Method
Isolation Forest
Evaluation Metrics
MAE
RMSE
MAPE
R² Score
Interactive Dashboard
Electricity trend visualization
Forecast comparison
Anomaly visualization
Model performance comparison
Dataset

Dataset Used:

UCI Individual Household Electric Power Consumption Dataset

Dataset Source:

UCI Machine Learning Repository Dataset Page

Direct Download:

Dataset Download Link

Problem Statement

Electricity consumption changes dynamically based on:

Human activity
Daily routines
Appliance usage
Seasonal effects
Unexpected events

The goal of this project is to:

Forecast future household electricity consumption.
Detect abnormal consumption behavior.
Compare multiple forecasting models.
Build an interactive dashboard for visualization and monitoring.
Project Architecture
Raw Dataset
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Forecasting Models
     ↓
Anomaly Detection
     ↓
Model Evaluation
     ↓
Streamlit Dashboard
Project Structure
electricity-forecasting-ai/
│
├── data/
│   ├── anomalies.csv
│   ├── model_comparison.csv
│   └── screenshots/
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── features.py
│   ├── split.py
│   ├── ml_models.py
│   ├── sarima_model.py
│   ├── deep_model.py
│   ├── evaluate.py
│   └── anomaly.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
Installation
1. Clone the Repository
git clone https://github.com/your-username/electricity-forecasting-ai.git
cd electricity-forecasting-ai
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Required Libraries
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
tensorflow
keras
streamlit
plotly
joblib
scipy
How to Run the Project
Step 1 — Download Dataset

Download dataset from:

Download Dataset

Extract and place:

household_power_consumption.txt

inside:

data/
Step 2 — Run Full Pipeline
python main.py

This will:

preprocess data
generate features
train models
evaluate performance
detect anomalies
save outputs
Step 3 — Launch Dashboard
streamlit run dashboard/app.py
Exploratory Data Analysis

The project performs multiple EDA tasks:

Full electricity consumption trend
Monthly average consumption
Hourly usage analysis
Day-of-week analysis
Correlation heatmaps
Distribution analysis
Feature Engineering

The project creates advanced time-series features including:

Time Features
Hour
Day
Month
Quarter
Weekday
Weekend flag
Lag Features
1-hour lag
24-hour lag
48-hour lag
Rolling Statistics
Rolling mean
Rolling standard deviation
Rolling max/min
Cyclical Encoding
Sine/Cosine transformations for hour and month
Forecasting Models
Random Forest

Tree-based ensemble model for nonlinear forecasting.

XGBoost

Gradient boosting model optimized for structured data.

SARIMA

Classical statistical time-series forecasting model.

LSTM

Deep learning recurrent neural network for sequential learning.

GRU

Efficient recurrent neural network similar to LSTM.

Anomaly Detection

The project identifies abnormal electricity usage using:

Z-Score Method

Detects statistical outliers.

Rolling Threshold

Flags points outside rolling mean ± standard deviation.

Isolation Forest

Machine learning-based anomaly detection.

Evaluation Metrics

The following metrics are used for model comparison:

Metric	Description
MAE	Mean Absolute Error
RMSE	Root Mean Squared Error
MAPE	Mean Absolute Percentage Error
R² Score	Coefficient of Determination
Dashboard Features

The Streamlit dashboard includes:

Overview
Electricity trends
KPI cards
Sub-metering analysis
EDA Section
Hourly analysis
Weekly patterns
Heatmaps
Forecasting Section
Actual vs predicted plots
Residual analysis
Performance metrics
Anomaly Detection
Interactive anomaly visualization
Method comparison
Model Comparison
RMSE comparison
MAE comparison
R² analysis
Radar chart
Dashboard Screenshots

Add screenshots here after running the dashboard.

Example:

![Dashboard](assets/dashboard.png)
Results

The project successfully:

Forecasts electricity consumption
Detects abnormal usage patterns
Compares multiple ML/DL models
Visualizes insights interactively

Expected best-performing models:

XGBoost for ML forecasting
LSTM for deep learning forecasting
Isolation Forest for anomaly detection
Future Improvements

Potential enhancements:

Real-time streaming data
Weather API integration
Smart home IoT integration
Energy-saving recommendations
Real-time alerts
Cloud deployment
Appliance-level prediction
Technologies Used
Category	Technologies
Programming	Python
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
ML Models	Scikit-learn, XGBoost
Deep Learning	TensorFlow, Keras
Statistical Models	Statsmodels
Dashboard	Streamlit
Learning Outcomes

This project demonstrates:

Time-series forecasting
Machine learning pipelines
Deep learning for sequential data
Anomaly detection
Feature engineering
Interactive dashboard development
End-to-end AI system design

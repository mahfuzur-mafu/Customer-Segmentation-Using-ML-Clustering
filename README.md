# Customer Segmentation Using ML Clustering

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This project implements customer segmentation using unsupervised machine learning clustering techniques, specifically KMeans, to group customers based on demographic and behavioral attributes. The segmented clusters help businesses tailor marketing strategies, improve customer engagement, and optimize resource allocation.

The project includes:
- A Jupyter Notebook (`project.ipynb`) for data exploration, preprocessing, model training, and visualization.
- A Streamlit web application (`app.py`) for interactive customer segment prediction.
- Sample dataset (`customer_segmentation.csv`) for training and demonstration.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Cluster Descriptions](#cluster-descriptions)
- [Results and Visualization](#results-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview
Customer segmentation divides a customer base into distinct groups sharing similar characteristics. This project uses KMeans clustering on features like age, income, spending behavior, recency, and purchase channels to identify 6 clusters. The model is trained on a retail customer dataset and deployed via a user-friendly Streamlit app where users can input customer details to predict segments in real-time.

Key goals:
- Identify high-value vs. low-engagement customers.
- Visualize clusters using PCA for better interpretability.
- Provide an interactive tool for segment prediction.

## Features
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding.
- **Clustering Model**: KMeans with optimal cluster selection (e.g., via elbow method or silhouette score).
- **Dimensionality Reduction**: PCA to reduce features for 2D visualization.
- **Model Persistence**: Saved KMeans model and scaler using Joblib for deployment.
- **Interactive App**: Streamlit-based UI to input customer data (age, income, spending, etc.) and get instant segment predictions.
- **Cluster Insights**: Pre-defined descriptions for each cluster to aid business interpretation.

## Dataset
The dataset (`customer_segmentation.csv`) contains 2,240 customer records with 29 features, including:
- **Demographics**: `Year_Birth` (used to calculate Age), `Education`, `Marital_Status`, `Income`.
- **Family**: `Kidhome`, `Teenhome`.
- **Behavioral**: `Recency` (days since last purchase), `MntWines`, `MntFruits`, etc. (spending on products), `NumWebPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`.
- **Campaign Responses**: `AcceptedCmp1` to `AcceptedCmp5`, `Response`.
- **Other**: `Dt_Customer` (enrollment date), `Complain`, etc.

Data source: Likely a public retail marketing dataset (e.g., similar to those on Kaggle). Note: The provided sample is truncated, but the full dataset is used in the notebook.

Example rows:
| ID   | Year_Birth | Education   | Marital_Status | Income | Kidhome | Teenhome | Dt_Customer | Recency | ... |
|------|------------|-------------|----------------|--------|---------|----------|-------------|---------|-----|
| 5524 | 1957      | Graduation | Single        | 58138 | 0      | 0       | 04-09-2012 | 58     | ... |
| 2174 | 1954      | Graduation | Single        | 46344 | 1      | 1       | 08-03-2014 | 38     | ... |

## Methodology
1. **Data Loading & Exploration**: Use Pandas for loading CSV, visualize distributions with Matplotlib/Seaborn.
2. **Preprocessing**:
   - Calculate `Age` from `Year_Birth`.
   - Derive `Total_Spending` as sum of product spends (e.g., `MntWines + MntFruits + ...`).
   - Handle missing values (e.g., impute income).
   - Scale features using StandardScaler.
3. **Feature Selection**: Focus on key features like `Age`, `Income`, `Total_Spending`, `NumWebPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `Recency`.
4. **Clustering**: Apply KMeans (n_clusters=6 based on analysis).
5. **Visualization**: Use PCA for 2D scatter plot of clusters.
6. **Model Saving**: Export KMeans model and scaler as `.pkl` files.
7. **Deployment**: Load saved model in Streamlit app for predictions.

Libraries used: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (KMeans, StandardScaler, PCA), Joblib.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/mahfuzur-mafu/Customer-Segmentation-Using-ML-Clustering.git
   cd Customer-Segmentation-Using-ML-Clustering
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (If `requirements.txt` is not present, install manually: `pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit`).

## Usage
### Running the Notebook
1. Open `project.ipynb` in Jupyter:
   ```
   jupyter notebook project.ipynb
   ```
2. Execute cells sequentially to train the model and generate visualizations.
3. The notebook saves `model_kmeans.pkl` and `scaler.pkl`.

### Running the Streamlit App
1. Launch the app:
   ```
   streamlit run app.py
   ```
2. Input customer details (Age, Income, Total Spending, Recency, Web Purchases, Store Purchases, Web Visits).
3. Click "Predict Segment" to view the predicted cluster and descriptions.

Example Input:
- Age: 35
- Income: 50000
- Total Spending: 1000
- Recency: 7
- Web Purchases: 10
- Store Purchases: 10
- Web Visits: 5

## Cluster Descriptions
Based on the trained model, the 6 clusters are interpreted as follows (from `app.py`):
| Cluster | Emoji | Description                          | Color Code |
|---------|-------|--------------------------------------|------------|
| 0      | 💼    | High budget, frequent web visitors   | #2196F3   |
| 1      | 💎    | High spenders with premium habits    | #9C27B0   |
| 2      | 🌐    | Frequent web visitors, moderate spenders | #FF9800 |
| 3      | 🏠    | Loyal store shoppers                 | #4CAF50   |
| 4      | 🛍    | Occasional shoppers with balanced habits | #FFC107 |
| 5      | 📉    | Low spenders, infrequent visits      | #F44336   |

These descriptions are displayed in the Streamlit app with color-coded cards.

## Results and Visualization

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/7344a484-8439-4e96-b502-6013dcfce397" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/0036ada0-e05e-4bff-84ef-c695810b4d54" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/c7bbd588-e4da-4ab8-852e-a10ce30f0fea" />

<img width="580" height="495" alt="image" src="https://github.com/user-attachments/assets/cbdc8a52-77c4-4198-986e-5e94c15ea3be" />

<img width="641" height="561" alt="image" src="https://github.com/user-attachments/assets/4a4b3996-c167-4368-9ad0-34f3537f6356" />

<img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/9773681b-79b2-422b-8bb6-3e12ca0c68b3" />



# Streamlit User Interface


- **Performance**: The model effectively groups customers; evaluate with silhouette score or inertia in the notebook for improvements.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


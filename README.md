# Singapore_flat_price_predictions

## Introduction
The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. 
There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. 
A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.

## Problem Statement 
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. 
This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

## Approach

### Data Collection and Preprocessing: 
Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. 
Preprocess the data to clean and structure it for machine learning.

### Feature Engineering: 
Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. 
Create any additional features that may enhance prediction accuracy.

### Model Selection and Training: 
Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, random forests, or XGB). 
Train the model on the historical data, using a portion of the dataset for training.

### Model Evaluation: 
Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.

### Streamlit Web Application: 
Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). 
Utilize the trained machine learning model to predict the resale price based on user inputs.

### Deployment on Render: 
Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.

### Testing and Validation: 
Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.

## Tools used
1. Python
2. Numpy
3. Pandas
4. Streamlit
5. Seaborn
6. Matplotlib
7. Sklearn
8. Render

## Key skills
1. Data preprocessing techniques
2. EDA
3. Machine learning techniques such as Regression
4. Hyper parameter tuning to optimize ML models
5. Web application using the Streamlit
6. Deployment 

## Streamlt Overview
![Predict Flat Prices](https://github.com/Sakthipavithran16/Singapore_flat_price_predictions/blob/main/Streamlit%20UI.JPG)

## Conclusion
The project will benefit both potential buyers and sellers in the Singapore housing market. 
Buyers can use the application to estimate resale prices and make informed decisions, while sellers can get an idea of their flat's potential market value. 
Additionally, the project demonstrates the practical application of machine learning in real estate and web development.

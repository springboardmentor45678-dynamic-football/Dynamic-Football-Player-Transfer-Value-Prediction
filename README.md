# Player Transfer Value Prediction System

## Overview

This repository contains an end-to-end machine learning project that predicts professional football players’ transfer market values by integrating player performance metrics, injury history, demographic attributes, and social media sentiment analysis.

The project demonstrates a complete AI workflow—from data ingestion and preprocessing to model development, evaluation, and deployment—following industry-aligned practices suitable for real-world sports analytics applications.


## Problem Statement

Player transfer valuation is influenced by multiple interacting factors such as on-field performance, availability, age, injuries, and public perception. Traditional valuation methods are often manual and subjective, leading to inconsistent outcomes.

This project addresses the problem by building a data-driven, machine learning–based system capable of producing reliable and scalable transfer value predictions.


## Data Sources

* Player performance statistics (goals, assists, minutes played, cards, clean sheets)
* Player profile data (age, height, nationality, position, preferred foot)
* Injury history (days missed, matches missed)
* Historical market value data
* Social media sentiment data derived from Twitter (VADER and TextBlob)

All datasets are sourced from publicly available repositories and processed in CSV format using Python.


## Data Engineering and Preprocessing

* Integrated multiple datasets using player ID, season, and standardized player names
* Handled missing values with domain-driven assumptions (e.g., zero injury days)
* Corrected invalid data types and rounded count-based features to integers
* Treated outliers using percentile-based capping to preserve data integrity
* Optimized memory usage with chunk-based data merging
* Engineered derived features such as player age
* Encoded categorical variables using label encoding and one-hot encoding
* Applied scaling and log transformations to stabilize model learning


## Machine Learning Models

The following regression models were implemented and evaluated:

* Linear Regression (baseline)
* Polynomial Regression
* Random Forest Regression
* XGBoost Regression

### Model Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Ensemble-based models significantly outperformed linear approaches. XGBoost delivered the best overall performance in terms of accuracy and generalization.


## Deployment Architecture

The project is deployed as a full-stack machine learning application with a clear separation of concerns.

### Backend (FastAPI)

* RESTful API for real-time predictions
* Input validation using data schemas
* Consistent preprocessing during training and inference
* High-performance and scalable architecture

### Frontend (Streamlit)

* Interactive web interface for entering player attributes
* Real-time display of predicted transfer values
* Seamless integration with the backend API


## Tech Stack

* Programming Language: Python
* Data Processing: Pandas, NumPy
* Machine Learning: Scikit-learn, XGBoost
* NLP and Sentiment Analysis: VADER, TextBlob
* Backend Framework: FastAPI
* Frontend Framework: Streamlit


## Project Highlights

* End-to-end machine learning system implementation
* Strong emphasis on data quality, preprocessing, and feature engineering
* Hands-on experience with ensemble learning techniques
* Production-oriented deployment using FastAPI and Streamlit
* Modular and scalable system design


## Use Cases

* Football clubs and scouting teams
* Sports analytics and performance evaluation
* AI-driven decision support systems in sports management


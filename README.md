
# Student Performance Prediction

## Overview
This project predicts a student's exam score based on study habits, attendance, sleep, and other factors.

## Problem Statement
Can we predict exam performance using student behavior and background features?

## Dataset
This project uses a student performance dataset with features such as:
- Hours Studied
- Attendance
- Sleep Hours
- Motivation Level
- Previous Scores
- Access to Resources

## Workflow
1. Loaded and explored the dataset
2. Handled missing values
3. Performed visualization and correlation analysis
4. Encoded categorical features
5. Split data into train and test sets
6. Trained a Linear Regression model
7. Evaluated using MAE and R² score

## Results
- MAE: 0.452
- R² Score: 0.769

## Key Insights
- Study hours have a strong effect on exam score
- Attendance also shows a useful relationship
- Some lifestyle factors influence performance

## Tech Stack
- Python
- Pandas
- Matplotlib
- Scikit-learn

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook/GoogleColab

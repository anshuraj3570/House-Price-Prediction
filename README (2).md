# House Price Prediction

A machine learning project for predicting house prices using property-related features.  
The notebook performs Exploratory Data Analysis (EDA), trains and compares multiple regression algorithms, selects the best-performing model using **R² score**, and performs additional analysis such as feature importance and PCA.

## Project Overview

This project uses a house-price dataset and applies several regression algorithms to predict the target variable:

* **Target:** `price`
* **Problem Type:** Regression
* **Evaluation Metric:** R² Score
* **Train/Test Split:** 90% training and 10% testing
* **Random State:** 2

## Dataset Features

The notebook uses features including:

* `bedrooms`
* `bathrooms`
* `sqft\_living`
* `sqft\_lot`
* `floors`
* `waterfront`
* `view`
* `condition`
* `grade`
* `sqft\_above`
* `sqft\_basement`
* `zipcode`
* `lat`
* `long`
* `sqft\_living15`
* `sqft\_lot15`
* Date-derived features:

  * `year`
  * `month`

The columns `id`, `price`, and the original `date` are removed from the training features after the date is converted into `year` and `month`.

## Project Workflow

```text
House Price Dataset
        ↓
Data Loading
        ↓
Exploratory Data Analysis
        ↓
Missing Value Check
        ↓
Feature Visualization \& Correlation Analysis
        ↓
Date Feature Engineering
        ↓
Train/Test Split
        ↓
Multiple Regression Models
        ↓
R² Score Comparison
        ↓
Best Model Selection
        ↓
Feature Importance
        ↓
Gradient Boosting Deviance Analysis
        ↓
PCA Analysis
```

## Exploratory Data Analysis

The notebook performs:

* Dataset information and data types analysis
* Statistical summary using `describe()`
* Missing-value checking
* Bedroom distribution analysis
* House-location visualization using latitude and longitude
* Price vs. living area analysis
* Price vs. latitude and longitude
* Bedrooms vs. price
* Total living area vs. price
* Waterfront vs. price
* Floors distribution and price relationship
* Condition vs. price
* Zipcode vs. price
* Correlation analysis
* Correlation heatmap

## Machine Learning Models

The following regression algorithms are trained and compared:

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **Gradient Boosting Regressor**

### Model Configuration

* Decision Tree: `max\_depth=8`
* Random Forest: `n\_estimators=300`, `max\_depth=12`
* Gradient Boosting:

  * `n\_estimators=400`
  * `max\_depth=5`
  * `learning\_rate=0.1`
  * `loss='squared\_error'`

The notebook selects the model with the highest R² score on the test set.

## Evaluation

The notebook creates an accuracy comparison table and chart using the R² score.

For regression, the project reports:

```text
R² Score
Accuracy % = R² × 100
```

The exact final accuracy and selected best algorithm are generated when the notebook is executed, because they depend on the dataset used.

## Additional Analysis

### 1\. Gradient Boosting Deviance

The notebook compares:

* Training Set Deviance
* Test Set Deviance

across boosting iterations using Mean Squared Error (MSE).

### 2\. Feature Importance

If the selected best model provides `feature\_importances\_`, the notebook displays the importance of each feature.

### 3\. PCA

Principal Component Analysis (PCA) is used as an optional dimensionality check to visualize cumulative explained variance.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model\_selection import train\_test\_split
from sklearn.linear\_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
```

## How to Run

### 1\. Open the Notebook

Open:

```text
House Price Prediction.ipynb
```

in Google Colab or Jupyter Notebook.

### 2\. Dataset Path

The notebook currently loads the dataset from:

```python
/content/drive/MyDrive/house price predication/house\_data.csv
```

Make sure the dataset is available at this location in Google Drive, or update the path in the notebook.

### 3\. Run All Cells

Run the notebook from top to bottom.

The notebook will:

* Load the dataset
* Perform EDA
* Prepare features
* Split the data
* Train six regression models
* Compare R² scores
* Select the best model
* Display additional model analysis

## Project Structure

```text
House-Price-Prediction/
│
├── House Price Prediction.ipynb
├── house\_data.csv
└── README.md
```

## Results

The notebook automatically prints a comparison similar to:

```text
Model                 R² Accuracy
---------------------------------
Linear Regression     ...
Ridge Regression      ...
Lasso Regression      ...
Decision Tree         ...
Random Forest         ...
Gradient Boosting     ...
```

The final selected model is the model with the highest test-set R² score.

## Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Cross-validation
* Additional regression metrics such as MAE and RMSE
* Better handling of categorical/location features
* Model serialization using Joblib/Pickle
* Building a Streamlit web application for interactive predictions

## Author

**Anshu Raj**

B.Tech — Computer Science / Data Science

\---

**Note:** The README describes the implementation present in the provided notebook. Exact model accuracy is intentionally not hard-coded because the notebook calculates it when executed.


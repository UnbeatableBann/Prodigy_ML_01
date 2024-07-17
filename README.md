# House Price Prediction Using Linear Regression

This repository contains code and resources for building a linear regression model to predict house prices based on square footage, the number of bedrooms, and the number of bathrooms. The aim of this project is to develop an accurate model that can assist in estimating the market value of a house given these features.

### Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

### Introduction
House price prediction is a common application of machine learning, particularly useful in the real estate industry. By analyzing the relationship between features like square footage, the number of bedrooms, and bathrooms, we can predict the selling price of a house. This project uses linear regression to build the predictive model.

### Dataset
The dataset used in this project contains information about houses, including:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms
- House Price (target variable)

The dataset is stored in a CSV file named `house_prices.csv`.

### Requirements
To run this project, you will need the following libraries:
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/UnbeatableBann/house-price-prediction.git
    cd house-price-prediction
    ```

2. Ensure the dataset `house_prices.csv` is in the repository directory.

3. Run the Jupyter notebook or Python script to train the model and make predictions:
    ```bash
    jupyter notebook house_price_prediction.ipynb
    ```
    or
    ```bash
    python house_price_prediction.py
    ```

### Model Training
The model training process involves:
1. Loading and exploring the dataset
2. Preprocessing the data (handling missing values, encoding categorical variables, etc.)
3. Splitting the data into training and testing sets
4. Training the linear regression model using the training set
5. Evaluating the model's performance on the testing set

### Evaluation
The model is evaluated using metrics such as:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) score

### Results
The results section in the notebook provides visualizations and performance metrics for the trained model. This helps in understanding the model's accuracy and areas of improvement.

### Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

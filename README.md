# TSLA Stock Price Prediction

This project aims to predict the closing prices of Tesla (TSLA) stock using various machine learning models. The project involves data preprocessing, feature selection, normalization, model training, and evaluation.

## Project Structure


- **data/**: Contains the dataset file.
- **notebooks/**: Contains the Jupyter notebook used for data analysis and model training.
- **models/**: Contains the trained models.
- **README.md**: Project documentation.
- **requirements.txt**: Python dependencies.

## Dataset

The dataset contains historical stock prices of Tesla (TSLA) including the following columns:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

## Project Steps

1. **Data Preprocessing**:
    - Load the dataset.
    - Convert the `Date` column to datetime format.
    - Set the `Date` column as the index.
    - Drop unnecessary columns.

2. **Feature Selection**:
    - Use Variance Threshold and Recursive Feature Elimination (RFE) to select important features.

3. **Normalization**:
    - Normalize the selected features using MinMaxScaler.

4. **Train-Test Split**:
    - Split the data into training and testing sets based on time series order.

5. **Modeling**:
    - Train three different models: Decision Tree Regressor, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).
    - Evaluate the models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) metrics.

## Results

- **Decision Tree Model**:
    - MSE: 0.005377281782433699
    - MAE: 0.05432291791765988
    - R2: 0.6374157512196461

- **SVM Model**:
    - MSE: 0.0027071041066865933
    - MAE: 0.04324210244602544
    - R2: 0.8174629211175671

- **KNN Model**:
    - MSE: 0.0027472214232536407
    - MAE: 0.04194821152828506
    - R2: 0.8147578541935937

The SVM model showed the best performance among the three models.

## Installation

To run this project, you need to have Python installed. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt

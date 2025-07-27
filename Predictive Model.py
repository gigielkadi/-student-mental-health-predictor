#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: gigielkadi
"""

import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QVBoxLayout, QLabel, QMessageBox, QComboBox
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Define a class for the data entry form GUI
class DataEntryForm(QWidget):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path  # Store the path to the dataset
        self.dataset = pd.read_csv(dataset_path)  # Load the dataset from the specified path
        self.setup_model()  # Call the function to set up the machine learning model
        self.initUI()  # Call the function to initialize the user interface

    def setup_model(self):
        # Define the column to predict and the columns to ignore in the model
        target = 'Do you have Anxiety?'  # Ensure this matches the exact column name for target
        drop_columns = ['Timestamp', 'Did you seek any specialist for a treatment?']

        # Prepare the feature matrix X by dropping the target and irrelevant columns
        X = self.dataset.drop(columns=drop_columns + [target])
        y = self.dataset[target]  # Define the target vector y

        # Identify categorical columns and create label encoders for them
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_cols}
        for col in self.categorical_cols:
            X[col] = self.label_encoders[col].fit_transform(X[col])  # Transform categorical data to numerical labels
        y = LabelEncoder().fit_transform(y)  # Transform the target variable

        # Create a pipeline for data preprocessing and logistic regression
        self.pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),  # Handle missing values by replacing them with the column mean
            StandardScaler(),  # Normalize features
            LogisticRegression(random_state=42)  # Logistic regression model
        )

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.pipeline.fit(X_train, y_train)  # Train the model using the training set

    def initUI(self):
        # Setup the window properties
        self.setWindowTitle('Data Entry for Mental Health Prediction')
        self.setGeometry(100, 100, 300, 600)  # Set the window position and size
        layout = QVBoxLayout()  # Use a vertical box layout

        # Create input fields for each feature
        self.inputs = {}
        for col in self.dataset.columns:
            if col not in ['Timestamp', 'Did you seek any specialist for a treatment?', 'Do you have Anxiety?']:
                layout.addWidget(QLabel(f"{col}:"))  # Add a label for each input field
                if col in self.categorical_cols:
                    combo = QComboBox(self)  # Create a dropdown for categorical fields
                    combo.addItems([''] + list(self.label_encoders[col].classes_))
                    self.inputs[col] = combo
                    layout.addWidget(combo)
                else:
                    le = QLineEdit(self)  # Create a text input for numeric fields
                    self.inputs[col] = le
                    layout.addWidget(le)

        # Add a button for making predictions
        btn_predict = QPushButton('Predict Anxiety', self)
        btn_predict.clicked.connect(self.predict_data)  # Connect the button click to the prediction function
        layout.addWidget(btn_predict)

        self.setLayout(layout)  # Set the layout of the window

    def predict_data(self):
        # Function to gather data from the inputs and make a prediction
        try:
            new_data = {}
            for col, widget in self.inputs.items():
                if col in self.categorical_cols and isinstance(widget, QComboBox):
                    value = widget.currentText()  # Get the selected value from the dropdown
                    if value:
                        new_data[col] = self.label_encoders[col].transform([value])[0]  # Transform the value to a label
                    else:
                        raise ValueError(f"Please select a value for {col}")
                else:
                    value = widget.text()  # Get the text input
                    if value:
                        new_data[col] = float(value)
                    else:
                        raise ValueError(f"Please enter a value for {col}")

            new_df = pd.DataFrame([new_data])  # Create a DataFrame from the input data
            pred = self.pipeline.predict(new_df)  # Predict using the model pipeline
            # Display the prediction result in a message box
            QMessageBox.information(self, 'Prediction Result', 'The prediction is: ' + ('Has Anxiety' if pred[0] == 1 else 'Does Not Have Anxiety'))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to predict data: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataEntryForm('/Users/gigielkadi/Desktop/Student Mental health.csv')
    ex.show()
    sys.exit(app.exec_())  # Start the application event loop

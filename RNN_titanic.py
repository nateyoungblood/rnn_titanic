#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Import necessary packages
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

# Load in dataset
train=pd.read_csv(r"C:\Users\natha\Downloads\train.csv")  

# Drop unnecessary variables
train=train.drop(["Name","Ticket","Cabin"],axis=1)

# Recode categorical variables to numeric
cat = {"Sex":     {"male": 0, "female": 1},
        "Embarked": {"S": 1, "C": 2, "Q": 3}}

train = train.replace(cat)

# Replace NaN values with 0
train['Age'] = train['Age'].fillna(0)
train['Embarked'] = train['Embarked'].fillna(0)

# Split data into dependent and independent variables
X=train.drop(["Survived"],axis=1)
y=train["Survived"]

# Split data into training and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = [9,]


# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
 )

# Calculate the Z-scores of each column in the training set:
train_df_mean = X_train.mean()
train_df_std = X_train.std()
train_df_norm = (X_train - train_df_mean)/train_df_std

# Calculate the Z-scores of each column in the test set.
test_df_mean = X_test.mean()
test_df_std = X_test.std()
test_df_norm = (X_test - test_df_mean)/test_df_std

# Compile the model using MAE loss function
model.compile(
    optimizer="adam",
    loss="mae",
)

# Fit model using training and test data
history = model.fit(
    train_df_norm, y_train,
    validation_data=(test_df_norm, y_test),
    batch_size=256,
    epochs=450,
)


# Evaluate model predictive accuracy using test data
results = model.evaluate(test_df_norm, y_test, batch_size=128)
print("test loss, test acc:", results)


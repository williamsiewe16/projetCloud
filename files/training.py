# Import libraries
import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Get the script arguments (regularization rate and training dataset ID)
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
args = parser.parse_args()

# Get the experiment run context
run = Run.get_context()

# Get the training dataset
print("Loading Data...")
df = run.input_datasets['training_data'].to_pandas_dataframe()

# Separate features and labels
y = df.valeur_fonciere
X = df.drop("valeur_fonciere",axis=1)

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
model = LinearRegression()
model.fit(X_train, y_train)

# calculate metrics
print('train score:', model.score(X_train, y_train))
print('test score:' , model.score(X_test, y_test))

run.log('train score', model.score(X_train, y_train))
run.log('test score', model.score(X_test, y_test))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/projetCloud_model.pkl')

run.complete()

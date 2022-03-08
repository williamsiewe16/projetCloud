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
parser.add_argument("--training-data", type=str, dest='training_data', help='training dataset')
parser.add_argument("--age", type=str, dest='age', help='age')
args = parser.parse_args()


# Get the experiment run context
run = Run.get_context()

training_data = args.training_data

# load the prepared data file in the training folder
print("Loading Data...")
file_path = os.path.join(training_data,'data.csv')
df = pd.read_csv(file_path)

run.log('final prep dataset len', len(df))
run.log('cols', len(df.columns))

# Separate features and labels
y = df.valeur_fonciere

run.log("nb null values", (df.isna()).sum(axis=0).sum())
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

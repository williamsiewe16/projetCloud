import json
import joblib
import numpy as np
import os

#Called when the service is loaded
def init():
    global model
    global scaler
    
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'projetCloud_model','2','projetCloud_model.pkl')
    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'projetCloud_scaler','2','myscaler.scl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    print(scaler.transform(data))
    print(model.coef_)
    print(model.intercept_)
    predictions = model.predict(scaler.transform(data))
    print(f"price: {predictions[0]}")
    # Return the predictions as JSON
    return json.dumps({"result": float(predictions[0])})

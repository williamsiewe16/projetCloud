from asyncio import TimerHandle
from cProfile import run
import streamlit as st
import pandas as pd
import numpy as np    
import requests
import json

def transform(val,min_,max_):
    return (val-min_)/(max_-min_)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def predict(data):


    type_local = {"Maison": 4, "Appartement": 3, 'Dépendance': 2, 'Local industriel. commercial ou assimilé': 1}

    # Nature mutation
    nature_mutation = {'Adjudication':0, 'Echange':1, 'Expropriation':2, 'Vente':3, 
    "Vente en l'état futur d'achèvement":4, 'Vente terrain à bâtir':5}

    b = pd.get_dummies(list(nature_mutation.keys()))
    b.iloc[0,:]=0
    b.iloc[0,nature_mutation[data["nature_mutation"]]]=1
    nat_mut = list(b.iloc[0,:].values)

    # Nature culture
    nature_culture = {'AB': 0,
    'AG': 1,'B': 2,'BF': 3,'BM': 4,'BP': 5,'BR': 6,'BS': 7,'BT': 8,'CA': 9,'CH': 10,'E': 11,'J': 12,
    'L': 13,'LB': 14,'P': 15,'PA': 16,'PC': 17,'PE': 18,'PH': 19,'PP': 20,'S': 21,'T': 22,'VE': 23,
    'VI': 24}

    b = pd.get_dummies(list(nature_culture.keys()))
    b.iloc[0,:]=0
    b.iloc[0,nature_culture[data["nature_culture"]]]=1
    nat_cul = list(b.iloc[0,:].values)


    input = [
        67.1,68.26,#137.02,116.6,88.1,
        data["nb_pieces_principales"],#np.random.randint(0,27,1)[0],  
        type_local[data["type_local"]],
        data["surface_terrain"],
        data["nb_pieces_principales"],
        data["surface_terrain"],
        data["longitude"],
        data["latitude"],
    ]
    input = input+nat_cul+nat_mut+[2020]

    url = "http://bb6d7e6b-5e0b-460d-92c6-cd2d874b0f52.westeurope.azurecontainer.io/score"
    #'http://www.ip-api.com/json'

    print(input)
    # Convert the array to a serializable list in a JSON document
    input_json = json.dumps({"data": [input]}, cls=NpEncoder)

    # Set the content type
    headers = { 'Content-Type':'application/json' }

    prediction = requests.post(url, input_json, headers = headers)
    prediction = json.loads(prediction.json())

    sidebar(2, value=prediction["result"])



def st_space(num=1):
    for i in range(num):
        st.write("")

def sidebar(running, value=4):
    if running == 0:
        st.sidebar.subheader("Remplissez tous les champs et soumettez le formulaire pour avoir une estimation de votre bien")

    elif running == 1:
        st.write("wait a moment...")
    
    else:
        st.sidebar.subheader("Notre modèle estime ce bien à:")
        st.sidebar.markdown('{:20,.2f} €'.format(abs(value)).strip())

def main():

    running = 0 

    sidebar(running)

    st.title('BEHLS Real Estate prediction App')
    st_space(2) 

    with st.form(key='my_form'):
        type_local = st.selectbox(
                "Type de local",
                ['Maison', 'Appartement', 'Dépendance',
            'Local industriel. commercial ou assimilé']
        )

        left_column, right_column = st.columns(2)

        with left_column:
            
            nb_pieces_principales = st.number_input('Nombre de pièces', min_value=1, max_value=15)
            nature_culture = st.selectbox(
            'Nature Culture',
            ['S', 'T', 'P', 'AB', 'J', 'BT', 'L', 'AG', 'VI', 'BR', 'VE', 'BS', 'PA',
        'B', 'E', 'BF', 'BP', 'PP', 'BM', 'PC', 'CA', 'LB', 'CH', 'PH', 'PE']
            )
            latitude = st.slider('Latitude?', 0, -100, 100)

        

        with right_column:

            nature_mutation=st.selectbox(
                    "Nature de la mutation",
                    ["Vente", "Vente en l'état futur d'achèvement", 'Echange', 'Adjudication', 'Vente terrain à bâtir', 'Expropriation']
            )
            
            surface_terrain = st.number_input('Surface du terrain', min_value=0.0, max_value=6000.0, step=2.0)

            longitude = st.slider('Longitude?', 0, -100, 100)

        
        
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            request = predict({
                "type_local": type_local,
                "nb_pieces_principales": nb_pieces_principales,
                "nature_culture": nature_culture,
                "nature_mutation": nature_mutation,
                "surface_terrain": surface_terrain,
                "latitude": latitude,
                "longitude": longitude
            })

            sidebar(1)


if __name__ == "__main__":
    main()

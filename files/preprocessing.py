# Import libraries
import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# Get the training dataset
print("Loading Data...")
df = run.input_datasets['raw_data'].to_pandas_dataframe()

run.log('raw_df_len', len(df))

# remove some useless columns
df.date_mutation=pd.to_datetime(df.date_mutation)
to_drop = ["id_mutation","numero_disposition","adresse_numero","adresse_nom_voie","adresse_code_voie","code_postal",
           "adresse_suffixe","code_commune","nom_commune","code_departement","ancien_code_commune", "ancien_nom_commune",
           "id_parcelle","ancien_id_parcelle","type_local","nature_culture","nature_culture_speciale","code_nature_culture_speciale",
          "lot1_numero","lot2_numero","lot3_numero","lot4_numero","lot5_numero", "numero_volume", "lot3_surface_carrez", "lot4_surface_carrez",
          "lot5_surface_carrez"]

reduced_df = df.drop(to_drop, axis=1)

# get_dummies
reduced_df = pd.get_dummies(reduced_df, columns=["code_nature_culture", "nature_mutation"])

# feature engineering
reduced_df["year_mutation"] = reduced_df.date_mutation.dt.year
reduced_df["code_type_local"] = 5-reduced_df.code_type_local

reduced_df = reduced_df.drop("date_mutation",axis=1)

# manage missing values
final_df = reduced_df.fillna(reduced_df.mean())


# Log raw row count
row_count = (len(final_df))
run.log('processed_rows', row_count)

# Normalization
X = final_df.drop("valeur_fonciere", axis=1)
cols = X.columns

scaler = MinMaxScaler()
final_df[cols] = scaler.fit_transform(X)

run.log_list('nulls', list((final_df.isna()).sum(axis=0).values))

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
final_df.to_csv(save_path, index=False, header=True)

joblib.dump(value=scaler, filename='outputs/myscaler.scl')
joblib.dump(value=final_df.columns, filename='outputs/cols.cl')

# End the run
run.complete()

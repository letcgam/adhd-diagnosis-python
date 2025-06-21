import os
import numpy as np
import pandas as pd
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_geometrical_features

path = 'tg-adhd-diagnosis/adhd-diagnosis-data/hrv_data/'
ibi_files = [file for file in os.listdir(path)]

all_ibi_data = []

for file_name in ibi_files:
    patient_id = file_name.replace('patient_hr_', '').replace('.csv', '')
    file_path = os.path.join(path, file_name)
    
    # Ler o arquivo, pulando as 2 primeiras linhas de metadados
    ibi_df = pd.read_csv(file_path, sep=';', skiprows=2, header=None)
    rr_intervals = ibi_df[1].tolist()

    time_domain_features = get_time_domain_features(rr_intervals)
    frequency_domain_features = get_frequency_domain_features(rr_intervals)
    geometrical_features = get_geometrical_features(rr_intervals)

    rri_object = {'ID': int(patient_id)}

    for col in time_domain_features.keys():
        rri_object[col] = time_domain_features[col]

    for col in frequency_domain_features.keys():
        rri_object[col] = frequency_domain_features[col]

    for col in geometrical_features.keys():
        rri_object[col] = geometrical_features[col]
    
    all_ibi_data.append(rri_object)

hrv_df = pd.DataFrame(all_ibi_data)
hrv_df = hrv_df.sort_values(by='ID')

hrv_df.to_csv(path + 'hrv_features.csv', index=False)
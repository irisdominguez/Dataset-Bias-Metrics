import os
USE_GPUS = '1'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = USE_GPUS
N_GPUS = len(USE_GPUS.split(','))

from fastai.vision.all import *
import sklearn

from fer import *
from fer.external_models import fairface

f = open(f'{LOGS_PATH}/demographic_analysis.log', 'a')
sys.stdout = f
sys.stderr = f
print = partial(print, flush=True)

datasets_to_process = source_dataset_configs

for name in datasets_to_process:
    print(f'Processing {name}')
    input_csv = f'{PROCESSED_PATH}/data/source_datasets/{name}.csv'
    output_csv = f'{PROCESSED_PATH}/data/source_datasets/{name}_demographic.csv'

    cropped_path = f'{DATA_PATH}/cropped/{name}/'
    df = pd.read_csv(input_csv)
    df = df[df['cropped_img'].notna()]
    imgs = cropped_path + df['cropped_img']

    fairface.predict_age_gender_race(output_csv, imgs)
    
for name in datasets_to_process:
    print(f'Merge csv: {name}')
    pathbase = f'{PROCESSED_PATH}/data/source_datasets/{name}.csv'
    dfbase = pd.read_csv(pathbase)
    
    pathdem = f'{PROCESSED_PATH}/data/source_datasets/{name}_demographic.csv'
    dfdem = pd.read_csv(pathdem)
    
    dfdem['face_name_align'] = dfdem['face_name_align'].apply(lambda x: x.split('/')[-1] if type(x) != float else x)
    
    df = pd.merge(dfbase, dfdem, how='outer', left_on='cropped_img', right_on='face_name_align')
    df = df.drop('face_name_align', axis=1)
    columns = list(df.columns)
    columns = columns[:-12] + [
        'age', 
        'age_conf_fair', 
        'age_scores_fair', 
        'gender', 
        'gender_conf_fair', 
        'gender_scores_fair', 
        'race',
        'race_conf_fair',
        'race_scores_fair',
        'race4',
        'race_conf_fair_4',
        'race_scores_fair_4'
    ]
    df = df[columns]
    
    path = f'{PROCESSED_PATH}/data/source_datasets/{name}_complete.csv'
    df.to_csv(path, index=None)
    
for name in datasets_to_process:
    print(f'Final csv: {name}')
   
    path = f'{PROCESSED_PATH}/data/source_datasets/{name}_complete.csv'
    df = pd.read_csv(path)
    df.insert(loc=0, column='dataset_name', value=name)
    df['cropped_img'] = df['dataset_name'] + '/' + df['cropped_img']
    pathf = f'{PROCESSED_PATH}/data/final/{name}.csv'
    df.to_csv(pathf, index=None)
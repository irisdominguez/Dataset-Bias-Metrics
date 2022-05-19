import os
USE_GPUS = '1'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = USE_GPUS
N_GPUS = len(USE_GPUS.split(','))

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pickle
import dlib
import pandas as pd
from tqdm.notebook import tqdm

from fastai.vision.all import *

from fer import *

import sys
sys.path.append('../')

from fer.external_models.facenet_pytorch import MTCNN, InceptionResnetV1
import fer.external_models.fairface as fairface


f = open(f'{LOGS_PATH}/prepare_and_crop.log', 'a')
sys.stdout = f
sys.stderr = f
print = partial(print, flush=True)

margin = 0.25
target_size = 224

cnn_face_detector, dlib_shape_predictor = fairface.load_face_models()

for name in source_dataset_configs:
    print(f'Process dataset: {name}')
    vocab = base_vocab
    if name == 'affectnet':
        vocab += ['contempt']
    ds = SourceDataset(name)
    path = f'{PROCESSED_PATH}/data/source_datasets/{name}.csv'
    df = ds.getPandas()

    cropped_path = f'{DATA_PATH}/cropped/{name}'
    ensure_dir(cropped_path)

    df['cropped_img'] = ''

    for i, image in df.iterrows():
        if i % (len(df) // 100) == 0:
            print(f'[{i}/{len(df)}]')
        img = dlib.load_rgb_image(os.path.join(ds.image_path, image['img_path']))
        face = fairface.get_single_face(img, size=target_size, padding=margin,
                                        cnn_face_detector=cnn_face_detector, 
                                        dlib_shape_predictor=dlib_shape_predictor)
        if face is not None:
            cropped_img_path = os.path.join(cropped_path, image['id'] + '.png')
            dlib.save_image(face, cropped_img_path)
            df.loc[i, 'cropped_img'] = image['id'] + '.png'


    df.to_csv(path, index=None)
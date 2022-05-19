## Original code from https://github.com/joojs/fairface

from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import argparse
import time

CNN_FACE_DETECTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dlib_models/mmod_human_face_detector.dat')
SHAPE_PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'dlib_models/shape_predictor_5_face_landmarks.dat')
FAIRFACE_PRETRAIN_PATH = os.path.join(os.path.dirname(__file__), 'fair_face_models/fairface_alldata_20191111.pt')
FAIRFACE_PRETRAIN_4RACE_PATH = os.path.join(os.path.dirname(__file__), 'fair_face_models/fairface_alldata_4race_20191111.pt')

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def detect_face(image_paths,  SAVE_DETECTED_AT, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTION_MODEL_PATH)
    sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    base = 2000  # largest width and height
    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            print('---%d/%d---' %(index, len(image_paths)))
        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(SAVE_DETECTED_AT,  path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            dlib.save_image(image, face_name)

def load_face_models():
    cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTION_MODEL_PATH)
    dlib_shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    
    return cnn_face_detector, dlib_shape_predictor
            
def get_single_face(img, default_max_size=800,size = 300, padding = 0.25, 
                    cnn_face_detector=None, dlib_shape_predictor=None):

    if cnn_face_detector == None:
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTION_MODEL_PATH)
    if dlib_shape_predictor == None:
        dlib_shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    
    base = 2000  # largest width and height

#     img = dlib.load_rgb_image(image_path)

    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)
    
    dets = cnn_face_detector(img, 1)
    
    num_faces = len(dets)
    if num_faces == 0:
        return None
    
    detection = dets[0]
    for d in dets:
        if d.confidence > detection.confidence:
            detection = d
        
    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    
    rect = detection.rect
    faces.append(dlib_shape_predictor(img, rect))
    images = dlib.get_face_chips(img, faces, size=size, padding = padding)
    
    # We only return the first image (there should only be one)
    return images[0]

def convertMillis(millis):
    seconds=int((millis/1000)%60)
    minutes=int((millis/(1000*60))%60)
    hours=int(millis/(1000*60*60))
    return f'{hours}:{minutes:02d}:{seconds:02d}'
    
def inclog(message, step, total, inittime):
    perc = step * 100 / total
    curtime = time.time()
    elapsed = (curtime - inittime) * 1000
    if perc > 0:
        remaining = elapsed / step * total
    else:
        remaining = 0
    elapsed = convertMillis(elapsed)
    remaining = convertMillis(remaining)
    print(f'{message} {perc:.0f}% [{step}/{total}] {elapsed} / {remaining}', flush=True)

def predict_age_gender_race(save_prediction_at, image_paths):
#     img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Predicting age, gender and race (device: {device})', flush=True)

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load(FAIRFACE_PRETRAIN_PATH))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load(FAIRFACE_PRETRAIN_4RACE_PATH))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_confs_fair = []
    gender_confs_fair = []
    age_confs_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []
    race_confs_fair_4 = []

    inittime = time.time()
    
    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            inclog("Predicting...", index, len(image_paths), inittime)

        face_names.append(image_path)
        image = dlib.load_rgb_image(image_path)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)
        
        race_conf = np.max(race_score)
        gender_conf = np.max(gender_score)
        age_conf = np.max(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)
        
        race_confs_fair.append(race_conf)
        gender_confs_fair.append(gender_conf)
        age_confs_fair.append(age_conf)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)
        race_conf = np.max(race_score)

        race_scores_fair_4.append(race_score)
        race_confs_fair_4.append(race_conf)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           race_preds_fair_4,
                           gender_preds_fair,
                           age_preds_fair,
                           race_confs_fair,
                           race_confs_fair_4,
                           gender_confs_fair,
                           age_confs_fair,
                           race_scores_fair, 
                           race_scores_fair_4,
                           gender_scores_fair,
                           age_scores_fair, ]).T
    
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_conf_fair',
                      'race_conf_fair_4',
                      'gender_conf_fair',
                      'age_conf_fair',
                      'race_scores_fair',
                      'race_scores_fair_4',
                      'gender_scores_fair',
                      'age_scores_fair']
    
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # race fair 4

    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    result[['face_name_align',
            'race',
            'race_conf_fair',
            'race4',
            'race_conf_fair_4',
            'gender',
            'gender_conf_fair',
            'age',
            'age_conf_fair',
            'race_scores_fair',
            'race_scores_fair_4',
            'gender_scores_fair',
            'age_scores_fair']].to_csv(save_prediction_at, index=False)

    print("saved results at ", save_prediction_at)
    
    return result


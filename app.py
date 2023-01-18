import cv2
import json
import numpy as np
from copy import copy

from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as ss

def insert_into_garment(face, garment):
    garment[0:face.shape[0], 0:face.shape[1]] = face
    return garment

def preprocess_images(face, garment, height = 128, width = 128):
    # face = cv2.imread(fr"{img_path}/facehair.png", cv2.IMREAD_ANYCOLOR)
    # garment = cv2.imread(fr"{img_path}/garment_top.png", cv2.IMREAD_ANYCOLOR)
    
    combined_image = insert_into_garment(face, garment)
    combined_image = cv2.resize(combined_image, (width, height))
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    combined_image = tf.convert_to_tensor(combined_image, dtype = tf.float32)
    
    preprocessed_image = tf.stack([combined_image], axis = 0)

    return preprocessed_image

def merged_image(face, garment, x, y):
    output_img = garment
    output_img[y: y + face.shape[0], x: x + face.shape[1]] = face
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    return output_img


ss.title("Auto CV")
ss.markdown("---")

col1, col2 = ss.beta_columns(2)

with col1:
    image1 = ss.file_uploader("Face image here")
    if image1 is not None:
        ss.image(image1)
        file_bytes = np.asarray(bytearray(image1.read()), dtype = np.uint8)
        image_1 = cv2.imdecode(file_bytes, 1)
        

with col2:
    image2 = ss.file_uploader("Garment image here")
    if image2 is not None:
        ss.image(image2)
        file_bytes = np.asarray(bytearray(image2.read()), dtype = np.uint8)
        image_2 = cv2.imdecode(file_bytes, 1)

join = ss.button("join")

if join:
    original_img_1, original_img_2 = copy(image_1), copy(image_2)
    preprocessed_image = preprocess_images(image_1, image_2)
    model = load_model("saved_models_inception/model_10/")
    y_preds = model.predict(preprocessed_image)
    out_image = merged_image(original_img_1, original_img_2, int(y_preds[0][0]), int(y_preds[0][1]))
    ss.markdown("---")
    ss.markdown("## combined image")
    ss.image(out_image)


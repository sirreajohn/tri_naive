
import cv2
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

def insert_into_garment(face, garment):
    garment[0:face.shape[0], 0:face.shape[1]] = face
    return garment

def preprocess_images(img_path, height = 128, width = 128):
    face = cv2.imread(fr"{img_path}/facehair.png", cv2.IMREAD_ANYCOLOR)
    garment = cv2.imread(fr"{img_path}/garment_top.png", cv2.IMREAD_ANYCOLOR)
    
    combined_image = insert_into_garment(face, garment)
    combined_image = cv2.resize(combined_image, (width, height))
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    combined_image = tf.convert_to_tensor(combined_image, dtype = tf.float32)
    
    preprocessed_image = tf.stack([combined_image], axis = 0)

    return preprocessed_image


def generate_merged_image(input_path, x, y):
    facehair_path = input_path + '/facehair.png'
    garment_top_path = input_path + '/garment_top.png'
    positions_path = input_path + '/positions.json'

    facehair_img = cv2.imread(facehair_path,cv2.IMREAD_UNCHANGED)
    garment_top_img = cv2.imread(garment_top_path,cv2.IMREAD_UNCHANGED)
    positions = json.load(open(positions_path))

    x_true = positions["x"]
    y_true = positions["y"]
    w = positions['w']
    h = positions['h']
    output_img = garment_top_img
    output_img[y: y + h, x: x + w] = facehair_img
    true_image = garment_top_img
    true_image[y_true: y_true + h, x_true: x_true + w]

    print(f"true_vals: x: {x_true}, y: {y_true}")
    print(f"pred_vals: x: {x}, y: {y}")

    cv2.imshow("predicted", output_img)
    cv2.waitKey(0)

    cv2.imshow("real", true_image)
    cv2.waitKey(0)



if __name__ == "__main__":
    model_path = "saved_models_inception\model_10"
    test_images_path = f"data/{np.random.randint(0, 3142)}"

    test_images_processed = preprocess_images(test_images_path)

    model = load_model(model_path)
    y_preds = model.predict(test_images_processed)

    generate_merged_image(test_images_path, int(y_preds[0][0]), int(y_preds[0][1]))


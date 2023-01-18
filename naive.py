import cv2
import json
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

class naive_approach:
    def __init__(self, image_size, file_path, learning_rate):
        self.image_size = image_size
        self.file_path = file_path
        self.learning_rate = learning_rate

    @staticmethod
    def insert_into_garment(face, garment):
        garment[0:face.shape[0], 0:face.shape[1]] = face
        return garment

    def create_dataset_tensorflow(self, test_split = 0.2, face_data_identifier = "facehair", garment_data_identifier = "garment_top"):

        data_dict = defaultdict(list)

        for image_set in glob(f"{self.file_path}/*"):  # i thought of making this o(n), but, it would mean disjoint update for each set of image data
            for input_data in glob(f"{image_set}/*"):  # I didnt want to update each set iteratively, instead went with 3 at once.

                if input_data.__contains__(face_data_identifier):  # one additional improvement would be to dynamically support different img formats
                    face_data = cv2.imread(input_data, cv2.IMREAD_ANYCOLOR)

                elif input_data.__contains__(garment_data_identifier):
                    garment_data = cv2.imread(input_data, cv2.IMREAD_ANYCOLOR)

                elif input_data.__contains__(".json"):
                    with open(input_data) as file:
                        pos_dict = json.load(file)

                else:
                    print(f"unknown_format / identifier : {input_data}")
                    continue

            combined_image = self.insert_into_garment(face_data, garment_data)
            combined_image = cv2.resize(combined_image, (self.image_size[0], self.image_size[1]))
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
            combined_image = tf.convert_to_tensor(combined_image, dtype = tf.float32)

            data_dict["image_data_train"].append(combined_image)
            data_dict["y_var_train"].append([pos_dict["x"], pos_dict["y"]])


        # data spliting
        split_indices = [int(len(data_dict["image_data_train"]) * test_split)]
        test_image, train_image = np.split(data_dict["image_data_train"], split_indices)
        test_y_vars, train_y_vars = np.split(data_dict["y_var_train"], split_indices)

        train_image = tf.stack(train_image, axis = 0)
        test_image = tf.stack(test_image, axis = 0)

        train_y_vars = np.array(train_y_vars)
        test_y_vars = np.array(test_y_vars)

        return train_image, train_y_vars, test_image, test_y_vars
    
    def get_model(self):
        inception_model_base = tf.keras.applications.inception_v3.InceptionV3(weights = "imagenet", include_top = False, input_shape = self.image_size)

        # adding custom top  
        out = inception_model_base.output
        out = Flatten()(out)

        x = Dense(6144, activation = "relu")(out)
        x = Dense(3072, activation = "relu")(x)
        pred = Dense(2, activation = "linear")(x)

        model = Model(inception_model_base.input, pred)

        # freeze weights
        for layer in inception_model_base.layers:
            layer.trainable = False

        model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = "mean_squared_error")

        return model

if __name__ == "__main__":

    IMAGE_SIZE = (128, 128, 3)
    FILE_PATH = "data"
    LEARNING_RATE = 1e-4
    EPOCHS = 10

    naive_obj = naive_approach(image_size = IMAGE_SIZE, file_path = FILE_PATH, learning_rate = LEARNING_RATE)
    train_image, train_y_vars, val_image, val_y_vars = naive_obj.create_dataset_tensorflow()
    
    model = naive_obj.get_model()
    reduce_leaning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.01, patience = 1, min_lr = 0.0001)
    history = model.fit(x = train_image, y = train_y_vars, epochs = EPOCHS, validation_data = (val_image, val_y_vars), callbacks = reduce_leaning_rate)

    model.save(f"saved_models_inception/model_{EPOCHS}")

    


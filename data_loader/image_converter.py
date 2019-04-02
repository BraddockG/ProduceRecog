import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.img import img_width,img_height

class ImageConverter:
    def __init__(self, destination):
        self.destination = destination

    def format_images(self, source):
        image_number = 0
        size = img_width, img_height

        for root, directories, filenames in os.walk(source):
            for filename in filenames:

                # This is here to ignore the ".DS_Store" file
                if filename.startswith('.'):
                    continue

                # Create directory if it doesn't exist
                target_directory = os.path.join(self.destination, os.path.basename(root))
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)

                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image.thumbnail(size, Image.ANTIALIAS)
                image = image.convert('RGB')
                image_file_name = str(image_number) + '.png'
                image.save(os.path.join(target_directory, image_file_name))

                image_number += 1

    def load_data(self):

        num_of_images = sum([len(files) for r, d, files in os.walk(self.destination)
                             if not files.__contains__(".DS_Store")])

        # One for each RGB channel
        NUMBER_OF_CHANNELS = 3

        X = np.ndarray(shape=(num_of_images, img_height, img_width, NUMBER_OF_CHANNELS), dtype=np.int)
        y = []

        i = 0
        for root, directories, filenames in os.walk(self.destination):
            for filename in filenames:

                # This is here to ignore the ".DS_Store" file
                if filename.startswith('.'):
                    continue

                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image_data = np.asarray(image)
                X[i] = image_data

                y.append(os.path.basename(root))

                i += 1

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)

        X_train = tf.keras.utils.normalize(X_train, axis=1)
        X_test = tf.keras.utils.normalize(X_train, axis=1)

        return (X_train, y_train), (X_test, y_test)

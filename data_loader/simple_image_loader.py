from base.base_data_loader import BaseDataLoader
from data_loader.image_converter import ImageConverter
import os


class SimpleImageLoader(BaseDataLoader):

    @staticmethod
    def load_data():
        destination = "data/interim"

        # This ignores the .DS_Store file
        lst = os.listdir(destination)
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')

        if not lst:
            image_converter = ImageConverter(destination)
            image_converter.format_images('data/raw')


    def __init__(self):
        self.load_data()


    def get_train_data(self):
        return 0 #self.X_train, self.y_train

    def get_test_data(self):
        return 0 #self.X_test, self.y_test

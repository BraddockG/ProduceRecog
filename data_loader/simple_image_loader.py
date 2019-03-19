from base.base_data_loader import BaseDataLoader
from data_loader.image_converter import ImageConverter
import os
from utils.img import img_width, img_height


class SimpleImageLoader(BaseDataLoader):

    @staticmethod
    def load_data():
        destination = "data/interim"

        # This ignores the .DS_Store file
        lst = os.listdir(destination)
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')

        image_converter = ImageConverter(destination)

        if not lst:
            image_converter.format_images('data/raw')

        return image_converter.load_data()

    def __init__(self, config):
        super(SimpleImageLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.load_data()
        self.X_train = self.X_train.reshape((-1, img_width, img_height))
        self.X_test = self.X_test.reshape((-1, img_width, img_height))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Flatten
from utils.img import img_width,img_height


class SimpleModel(BaseModel):
    def __init__(self, config):
        super(SimpleModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(img_height, img_width)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['accuracy'],
        )

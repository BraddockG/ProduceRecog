from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential

from base.base_model import BaseModel


class TutorialModel(BaseModel):
    def __init__(self, config):
        super(TutorialModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 60, 80), data_format='channels_first'))
        self.model.add(Convolution2D(32, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='softmax'))
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
        )

from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten


class SimpleModel(BaseModel):
    def __init__(self, config):
        super(SimpleModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(60, 80)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='softmax'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
        )

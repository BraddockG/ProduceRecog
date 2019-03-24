from comet_ml import Experiment
from data_loader.simple_image_loader import SimpleImageLoader
from models.simple_model import SimpleModel
from models.tutorial_model import TutorialModel
from trainers.simple_trainer import SimpleModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs


def main():
    config = process_config("configs/simple_config.json")

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = SimpleImageLoader(config)

    print('Create the model.')
    model = SimpleModel(config)

    print('Create the trainer')
    trainer = SimpleModelTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()

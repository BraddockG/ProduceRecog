from data_loader.simple_image_loader import SimpleImageLoader

def main():
    print('Create the data generator.')
    data_loader = SimpleImageLoader()

    print('Create the model.')
    model = SimpleMnistModel(config)


if __name__ == '__main__':
    main()

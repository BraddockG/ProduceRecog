from PIL import Image
import os

class ImageConverter:
    def __init__(self, destination):
        self.destination = destination

    def format_images(self, source):
        image_number = 0
        size = 30, 40

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
                image = image.convert('LA')
                image_file_name = str(image_number) + '.png'
                image.save(os.path.join(target_directory, image_file_name))

                image_number += 1


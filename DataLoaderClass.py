import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import  DataLoader
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

class DataLoaderClass:
    def __init__(self, x_paths, y_labels, batch_size, input_shape, shuffle=True):
        unique_ids = list(set(y_labels))
        self.id_to_numeric = {id: i for i, id in enumerate(unique_ids)}
        self.x_paths = x_paths
        self.y_labels = [self.id_to_numeric[label] for label in y_labels]
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.on_epoch_end()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.x_paths) // self.batch_size

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            indexes = [indexes]

        x_batch = [self.load_and_preprocess_image(self.x_paths[i]) for i in indexes]
        if self.y_labels is not None:
            y_batch = [self.y_labels[i] for i in indexes]
            return x_batch, y_batch
        else:
            return x_batch
        
        #x_batch = [self.load_and_preprocess_image(self.x_paths[i]) for i in indexes]
        #y_batch = [self.y_labels[i] for i in indexes]
        #x_batch = [x.cpu().numpy() for x in x_batch]
        #y_batch = np.array(y_batch)

        #return np.array(x_batch), y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_and_preprocess_image(self, image_path):
        image = np.load(image_path)
        image = image.astype('float32')
        # image = some_resize_function(image, self.input_shape)
        image_tensor = torch.tensor(image).to(self.device)
        return image_tensor
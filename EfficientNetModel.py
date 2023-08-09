import timm
import tensorflow as tf
import torch
from tensorflow.keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from tensorflow.keras import layers, Model
from torch import nn
from efficientnet_pytorch import EfficientNet

class EfficientNetModel:
    def __init__(self, input_shape, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = torch.nn.Linear(self.efficientnet._fc.in_features, num_classes)
        self.number_classes = num_classes
        self.model = self.create_model()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return x

    def get_parameters(self):
        return list(self.efficientnet.parameters()) + list(self.fc.parameters())

    def create_model(self):
        model = timm.create_model('efficientnet_b0', pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.number_classes),
            nn.Sigmoid()
        )
        return model

    def train(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=10):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = load_model(path)

    def predict(self, x):
        return self.model.predict(x)
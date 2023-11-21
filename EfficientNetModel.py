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
from DataLoaderClass import DataLoaderClass

class EfficientNetModel:
    def __init__(self, input_shape, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = torch.nn.Linear(self.efficientnet._fc.in_features, num_classes)
        self.number_classes = num_classes
        self.model = self.create_model()
        self.input_shape = input_shape

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return x

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())


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
        # Criar DataLoader ou adaptar conforme necessário
        train_loader = DataLoaderClass(x_train, y_train, batch_size=batch_size, input_shape=self.input_shape)
        val_loader = DataLoaderClass(x_val, y_val, batch_size=batch_size, input_shape=self.input_shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        optimizer = Adam(self.get_parameters())
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss}')

            # Lógica de validação
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    
                    val_outputs = self.model(val_inputs)
                    val_loss += criterion(val_outputs, val_labels.unsqueeze(1).float()).item()

            val_loss /= len(val_loader)
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

            # Adicione lógica adicional, como salvar o modelo se a perda de validação diminuir, etc.
            # Exemplo: Salvando o modelo se a perda de validação diminuir
            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Saved the best model!")

        print("Training complete.")
        # Criar DataLoader ou adaptar conforme necessário
        train_loader = DataLoaderClass(x_train, y_train, batch_size=batch_size, input_shape=self.input_shape)
        val_loader = DataLoaderClass(x_val, y_val, batch_size=batch_size, input_shape=self.input_shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        optimizer = Adam(self.get_parameters())
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f'Epoch {epoch + 1}, Loss: {train_loss}')

    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = load_model(path)

    def predict(self, x):
        return self.model.predict(x)
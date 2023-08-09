import tensorflow as tf
import torch
from tensorflow.keras.applications import EfficientNetB0
from torch.cuda.amp import GradScaler, autocast
from Plotter import Plotter
from tensorflow.keras.models import Model

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, loss_function, optimizer, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(self, num_epochs=10, plot_losses=True):
        train_losses = []
        valid_losses = []
        scaler = GradScaler()

        for epoch in range(num_epochs):
            self.model.train()

            for inputs, labels in self.train_loader:
                inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                labels = torch.tensor(labels).to(self.device)
                print(labels)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels.unsqueeze(1).float())
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            valid_loss = self.validate()
            valid_losses.append(valid_loss)

            if plot_losses:
                Plotter.plot_loss_graph(train_losses, valid_losses, epoch + 1)

        return train_losses, valid_losses

    def validate(self):
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels.unsqueeze(1).float())

                valid_loss += loss.item()

        valid_loss /= len(self.val_loader)
        return valid_loss
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split as sk_train_test_split
from EfficientNetModel import EfficientNetModel
from DataLoaderClass import DataLoaderClass

x_paths = []
csv_file_path = './data/training_labels.csv'
df = pd.read_csv(csv_file_path)

y_labels = df['id'].values
data_root = './data'

for level1 in range(16):

    for level2 in range(16):

        for level3 in range(16):
            folder_path = os.path.join(data_root, 'train', hex(level1)[2], hex(level2)[2], hex(level3)[2])

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                x_paths.append(file_path)

x_train, x_val, y_train, y_val = sk_train_test_split(x_paths, y_labels, test_size=0.2, random_state=42)

input_shape = (224, 224, 3)
train_loader = DataLoaderClass(x_train, y_train, batch_size=32, input_shape=input_shape)
val_loader = DataLoaderClass(x_val, y_val, batch_size=32, input_shape=input_shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNetModel(input_shape=input_shape, num_classes=1)

optimizer = Adam(model.get_parameters())
criterion = nn.BCELoss()

for epoch in range(10):
    model.train(x_train, y_train, x_val, y_val)
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}')

torch.save(model.state_dict(), 'path/to/model.pth')

model.load_state_dict(torch.load('path/to/model.pth', map_location=device))
model.eval()

x_test = []

for level1 in range(16):
    
    for level2 in range(16):

        for level3 in range(16):
            folder_path = os.path.join(data_root, 'test', hex(level1)[2], hex(level2)[2], hex(level3)[2])
            
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                x_test.append(file_path)

x_test_loader = DataLoaderClass(x_test, None, batch_size=32, input_shape=input_shape, shuffle=False)
predictions = model.predict(x_test_loader)

def visualize_results(x_test, predictions, threshold=0.5):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Exemplos de Predição", fontsize=16, y=1.05)
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_test[i], cmap='gray')
        ax.set_title("Normal" if predictions[i] < threshold else "Pneumonia", fontsize=12, weight='bold', y=-0.2)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

visualize_results(x_test, predictions)
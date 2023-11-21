import pandas as pd
import numpy as np
import os

data_dir = 'G:/Pasta Pessoal/Projetos/gravitational-wave-detection/data'
training_labels_dir = f'{data_dir}/training_labels.csv'
sample_submission_dir = f'{data_dir}/sample_submission.csv'
test_dir = f'{data_dir}/test'
train_dir = f'{data_dir}/train'
level_dir = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
test_array = np.array([])
train_array = np.array([])

training_labels = pd.read_csv(training_labels_dir)
sample_submission = pd.read_csv(sample_submission_dir)

for level_dir1 in level_dir:
    for level_dir2 in level_dir:
        for level_dir3 in level_dir:
            folder_path_train = os.path.join(data_dir, 'train',level_dir1,level_dir2,level_dir3)
            folder_path_test = os.path.join(data_dir, 'test',level_dir1,level_dir2,level_dir3)

            for file_name_train in os.listdir(folder_path_train):
                file_path = f'{test_dir}/{level_dir1}/{level_dir2}/{level_dir3}/{file_name_train}'
                if os.path.exists(file_path):
                    np.append(train_array, np.load(file_path))

            for file_name_test in os.listdir(folder_path_test):
                file_path = f'{test_dir}/{level_dir1}/{level_dir2}/{level_dir3}/{file_name_test}'
                if os.path.exists(file_path):
                    np.append(test_array, np.load(file_path))

print(training_labels.head())
print(sample_submission.head())
print(train_array)
print(test_array)
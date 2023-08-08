'''
Holds helper functions for train.py
'''
import pickle
import math
import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import numpy as np


# Define a mapping dictionary to map labels to integers
label_to_int = {
    'CH': 0,
    'CU': 1,
    'EP': 2,
    'FC': 3,
    'FF': 4,
    'FS': 5,
    'FT': 6,
    'KC': 7,
    'KN': 8,
    'SC': 9,
    'SI': 10,
    'SL': 11,
}


# Define a custom dataset to load data from the pickle file
class PitchDataset(Dataset):
    def __init__(self, pickle_file, isTrain=True):
        extension = '_train.pkl' if isTrain else '_test.pkl'
        pickle_file += extension
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        # Map labels to integers
        self.data['pitch_type'] = self.data['pitch_type'].map(label_to_int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data.iloc[idx, :-1].values
        label = self.data.iloc[idx, -1]
        if math.isnan(label):
            print(inputs, label)
        labels = torch.tensor(int(label), device="cuda")
        labels = torch.nn.functional.one_hot(labels, num_classes=16)

        return inputs, label

# Calculate confusion with sklearn confusion_matrix
def calculate_confusion_matrix(all_predictions, all_labels):
    # Convert predictions and labels to numpy arrays
    predictions_np = all_predictions.cpu().numpy()
    labels_np = all_labels.cpu().numpy()

    # Get the predicted class for each sample
    predicted_classes = np.argmax(predictions_np, axis=1)
    print("Predicted Classes:", predicted_classes)
    print("Actual Labels:", labels_np[:10])
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(labels_np, predicted_classes)

    return conf_matrix

# Inverse mapping dictionary to retrieve the original label from the integer
int_to_label = {v: k for k, v in label_to_int.items()}
class_labels = [label for label, _ in sorted(label_to_int.items(), key=lambda x: x[1])]

# Create, format, and print the confusion matrix
def print_confusion_matrix(all_predictions, all_labels):
    # Calculate the confusion matrix
    conf_matrix = calculate_confusion_matrix(all_predictions, all_labels)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print("   " + "  ".join(class_labels))
    for i, row in enumerate(conf_matrix):
        print(f"{class_labels[i]} {row}")

# Calculate accuracy for an epoch
def calculate_accuracy(output, target):
    batch_size = target.shape[0]

    _, target_classes = torch.max(target, dim=-1)
    _, output_classes = torch.max(output, dim=-1)


    matching = output_classes.eq(target_classes)
    num_correct = matching.long().sum().item()

    accuracy = float(num_correct) / batch_size
    return accuracy
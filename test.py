'''
Script to perform various analyses on trained model
Currently implemented confusion matrix and working on t-SNE
reference classes:
    'CH': 0 : Changeup
    'CU': 1 : Curveball
    'EP': 2 : Eephus
    'FC': 3 : Cutter
    'FF': 4 : Four-seam Fastball
    'FS': 5 : Splitter
    'FT': 6 : Two-seam Fastball
    'KC': 7 : Knuckle Curve
    'KN': 8 : Knuckleball
    'SC': 9 : Screwball
    'SI': 10 : Sinker
    'SL': 11 : Slider
'''
import yaml
import argparse
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from models.pid_linear import PitchIdentifierLinear  # Update the model import as needed
from train import CustomDataset

parser = argparse.ArgumentParser(description='MLB-DeepLearning')
parser.add_argument('--config', default='./config.yaml')

def load_model(model_path):
    model = PitchIdentifierLinear()  # Replace with your actual model class
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_classification(model, test_loader):
    class_names = {
        0: 'Changeup', 1: 'Curveball', 2: 'Eephus', 3: 'Cutter',
        4: 'Four-seam Fastball', 5: 'Splitter', 6: 'Two-seam Fastball',
        7: 'Knuckle Curve', 8: 'Knuckleball', 9: 'Screwball',
        10: 'Sinker', 11: 'Slider'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Perform forward pass
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted_classes.cpu().numpy())

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(y_true, y_pred, target_names=class_names.values())

    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, columns=class_names.values(), index=class_names.values()))
    print("\nClassification Report:")
    print(class_report)

# currently bugged!
def visualize_tSNE(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    embeddings = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            # Get the output embeddings of your model
            outputs = model.get_embeddings(inputs)
            embeddings.extend(outputs.cpu().numpy())
            targets.extend(labels.numpy())

    # Perform t-SNE to reduce the embeddings to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a DataFrame with the t-SNE results and class labels
    tsne_df = pd.DataFrame(data=embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['Class'] = targets

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Class', data=tsne_df, palette='tab20')
    plt.title('t-SNE Visualization of MLB Pitch Descriptions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def main():
    #args from yaml file
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # set args object
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # Load the model
    model_path = 'checkpoints/PID_Linear_trained.pt'
    model = load_model(model_path)

    # Load the data loaders
    dataset = CustomDataset(args.dataset)
    train_size = int(args.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Call the evaluation function
    evaluate_classification(model, test_loader)

    # Call the t-SNE visualization function
    visualize_tSNE(model, test_loader)

if __name__ == "__main__":
    main()
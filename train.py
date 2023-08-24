'''
    Train loop that accepts arguments from .yaml files in /models
    Implemented in PyTorch reading from .pickle files for data
    Execute with: python train.py --configs configs/configs_file.yaml
'''
# tools
import os
import yaml
import argparse
import matplotlib.pyplot as plt
from utils import PitchDataset, print_confusion_matrix, calculate_accuracy

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# models
from models.pid_transformer import PitchIdentifierTransformer
from models.pid_linear import PitchIdentifierLinear

parser = argparse.ArgumentParser(description='MLB-DeepLearning')
parser.add_argument('--config', default='./config.yaml')

def main():
    #args from yaml file
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    # set args object
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    # Split the dataset into train and test sets
    train_dataset = PitchDataset(args.dataset, True)
    test_dataset = PitchDataset(args.dataset, False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Set up the model and move it to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")
    if args.model == "PID_Transformer":
        model = PitchIdentifierTransformer(d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.dim_feedforward, dropout=args.dropout, num_layers=args.num_layers, device=device)
    elif args.model == "PID_Linear":
        model = PitchIdentifierLinear(hidden_dim=args.hidden_dim, dropout=args.dropout, device=device)
    
    model.to(device)

    # Define loss function and optimizer
    if args.criterion == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid criterion")
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise Exception("Invalid optimizer")

    #tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # Training loop
    for epoch in range(args.epochs):
        batch_losses = []
        batch_accuracies = []
        model.train()
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()          

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_accuracy = calculate_accuracy(outputs, labels)

            batch_losses.append(loss.item())
            batch_accuracies.append(batch_accuracy)

            if idx % 100 == 0:
                print(f"{idx} : {loss.item()}")

        avg_train_loss = sum(batch_losses) / len(batch_losses)
        avg_train_acc = sum(batch_accuracies) / len(batch_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        val_batch_losses = []
        val_batch_accuracies = []
        all_predictions = None
        all_labels = None
        model.eval()
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)  # Assuming the last column is the target label
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                batch_accuracy = calculate_accuracy(outputs, labels)

                val_batch_losses.append(loss.item())
                val_batch_accuracies.append(batch_accuracy)
                all_predictions = outputs if all_predictions is None else torch.cat((all_predictions, outputs), 0)
                all_labels = labels if all_labels is None else torch.cat((all_labels, labels), 0)
                if idx % 100 == 0:
                    print(f"{idx} : {loss.item()}")

        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        avg_val_acc = sum(val_batch_accuracies) / len(val_batch_accuracies)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        print(f"--------EPOCH {epoch+1}, TRAIN LOSS: {avg_train_loss}, TRAIN ACC: {avg_train_acc}, VAL LOSS: {avg_val_loss}, VAL ACC: {avg_val_acc}---------")
        print_confusion_matrix(all_predictions, all_labels)
        print("---------------------------------------------------")

    # Save the trained model
    model_path = os.path.join('checkpoints', f"{args.model}_trained.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to '{model_path}'.")

    # Plots
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("figures\\loss.png")
    plt.show()
    plt.plot(train_accuracies, label="Training Accuracies")
    plt.plot(val_accuracies, label="Validation Accuracies")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("figures\\acc.png")
    plt.show()

if __name__ == "__main__":
    main()

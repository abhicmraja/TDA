import torch
import torch.optim as optim
import argparse
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

# from BDB_MLP_Dataloader import *
from New_Dataloader import *
from model_TRIO import *


def loss_function(CNN_output, label):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(CNN_output, label)
    return loss


# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", ncols=100):
            inputs, one_hots, mlps, labels = data
            inputs, one_hots, mlps, labels = inputs.to(device), one_hots.to(device), mlps.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs, one_hots, mlps)
            preds = outputs.cpu().numpy()

            # Predicted and true categories
            pred_category = np.argmax(preds, axis=1)
            true_category = np.argmax(labels.cpu().numpy(), axis=1)

            # Compare predictions and ground truth
            total_correct += np.sum(pred_category == true_category)
            total_samples += len(true_category)

            # Store for metrics
            all_preds.extend(preds)
            all_labels.extend(true_category)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metrics calculation
    if all_preds.ndim == 1:
        all_preds = np.expand_dims(all_preds, axis=1)  # Convert to 2D if necessary

    # Precision, Recall, F1 - Using argmax for multi-class predictions
    predicted_classes = np.argmax(all_preds, axis=1)

    precision = precision_score(all_labels, predicted_classes, average='macro', zero_division=0)
    recall = recall_score(all_labels, predicted_classes, average='macro', zero_division=0)
    f1 = f1_score(all_labels, predicted_classes, average='macro', zero_division=0)

    # Multi-class average precision score (precision-recall curve for each class)
    # For multi-class, average_precision_score expects the label probabilities (not just the predictions)
    mAP = np.mean([average_precision_score(all_labels == i, all_preds[:, i]) for i in range(all_preds.shape[1])])

    # Accuracy
    accuracy = np.sum(predicted_classes == all_labels) / len(all_labels)

    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print(f"mAP: {mAP:.5f}")
    print(f"Accuracy: {accuracy:.5f}")

    return precision, recall, f1, mAP, accuracy


# Training function
def train(model, train_dataloader, val_dataloader, loss_function, num_epochs, checkpoint_dir="checkpoints", log_file="training_log.txt", learning_rate=0.001, save_every=5):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize optimizer and device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print()
    print("Using device: ", device)
    model.to(device)
    print()
    time.sleep(1)

    # Open log file for writing
    with open(log_file, 'w') as log:
        log.write("Epoch,Loss,Precision,Recall,F1,mAP,Accuracy\n")  # Write CSV header

        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0

            # Add tqdm progress bar to the dataloader loop
            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as progress_bar:
                for inputs, one_hots, mlps, labels in progress_bar:
                    # Move data to device
                    inputs, one_hots, mlps, labels = inputs.to(device), one_hots.to(device), mlps.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # Forward pass
                    assert torch.isfinite(inputs).all(), f"Input contains NaN or Inf values {inputs}"
                    assert torch.isfinite(one_hots).all(), f"OneHot contains NaN or Inf values {one_hots}"
                    assert torch.isfinite(mlps).all(), f"MLP contains NaN or Inf values{mlps}"

                    CNN_output = model(inputs, one_hots, mlps)
                    if torch.isnan(CNN_output).any():
                        assert False, f"NaN detected in output"

                    # Compute loss
                    loss = loss_function(CNN_output, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Update progress bar with the current loss
                    progress_bar.set_postfix(loss=loss.item())

            # Compute average loss for the epoch
            epoch_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            # Run evaluation every `save_every` epochs
            if (epoch + 1) % save_every == 0:
                # Save model checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

                print(f"Running evaluation after epoch {epoch + 1}...")
                precision, recall, f1, mAP, accuracy = evaluate(model, val_dataloader, device)

                # Log to file
                log.write(f"{epoch + 1},{epoch_loss:.4f},{precision:.5f},{recall:.5f},{f1:.5f},{mAP:.5f},{accuracy:.5f}\n")

        print("Training complete.")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train a model for NFL BDB 2025")

    # Add arguments
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size of the training data')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Path to the training data')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Path to the validation data')
    parser.add_argument("--output_nodes", type=int, default=2, help="Number of output nodes of the model")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--save_every', type=int, default=5, help='Save model and evaluate every N epochs')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='Path to the log file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    # Parse arguments
    args = parser.parse_args()

    # Set up your data loaders
    train_dataset = BDB_MLP_Dataset(args.train_data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = BDB_MLP_Dataset(args.val_data_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = TRIO()

    # Train the model with parsed arguments
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        log_file=args.log_file,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    main()

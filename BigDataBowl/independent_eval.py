import torch
from tqdm import tqdm
from BDB_MLP_Dataloader import *
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from model_TRIO import TRIO
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

val_dataset = BDB_MLP_Dataset("/Users/abhicmraja/Python Projects/BigDataBowl/Dataset/Test")
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

state_dict = torch.load("/Users/abhicmraja/Python Projects/BigDataBowl/checkpoints/model_epoch_1.pth")

model = TRIO()
model.load_state_dict(state_dict)
model.to(device)

model.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []
total_correct = 0
total_samples = 0


with torch.no_grad():
    for data in tqdm(val_dataloader, desc="Evaluating", ncols=100):
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
    assert False, "something wrong"

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

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def second_prediction_metrics(y_true, y_pred_probs):
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # Get second-highest prediction
    sorted_indices = np.argsort(y_pred_probs, axis=1)
    second_preds = sorted_indices[:, -2]  # Index of the second-highest prediction

    # Compute MAP for the second prediction
    map_score = 0.0
    for i in range(len(y_true)):
        if y_true[i] == second_preds[i]:
            map_score += 1.0  # Precision at rank 2 is 1/2
    second_map = map_score / len(y_true)

    # Compute binary metrics (precision, recall, F1-score)
    y_pred_binary = (second_preds == y_true).astype(int)  # 1 if second prediction matches true label, else 0
    y_true_binary = np.ones_like(y_pred_binary)  # Ground truth is always 1 for correct predictions

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    metrics = {
        'map': second_map,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return metrics


print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1 Score: {f1:.5f}")
print(f"mAP: {mAP:.5f}")
print(f"Accuracy: {accuracy:.5f}")

metrics = second_prediction_metrics(all_labels, all_preds)
print(f"Top-3 MAP: {metrics['map']:.5f}")
print(f"Top-3 Precision: {metrics['precision']:.5f}")
print(f"Top-3 Recall: {metrics['recall']:.5f}")
print(f"Top-3 F1: {metrics['f1']:.5f}")
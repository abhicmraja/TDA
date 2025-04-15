from torch.utils.data import TensorDataset
from New_Dataloader import *
from tqdm import tqdm


def weed_out_problematic_inputs(dataloader):
    valid_inputs = []
    valid_labels = []
    valid_one_hots = []
    valid_mlps = []

    total_invalid_count = 0

    # Use tqdm to track progress
    for data in tqdm(dataloader, desc="Processing Batches"):
        # Check if all inputs and targets are finite
        inputs, one_hots, mlps, labels = data
        inputs_valid = torch.isfinite(inputs).all(dim=tuple(range(1, inputs.ndim)))  # [batch_size]
        labels_valid = torch.isfinite(labels).all(dim=tuple(range(1, labels.ndim)))  # [batch_size]
        one_hots_valid = torch.isfinite(one_hots).all(dim=tuple(range(1, one_hots.ndim)))  # [batch_size]
        mlps_valid = torch.isfinite(mlps).all(dim=tuple(range(1, mlps.ndim)))  # [batch_size]

        # Step 2: Combine masks along the batch dimension
        valid_mask = inputs_valid & labels_valid & one_hots_valid & mlps_valid  # [batch_size]
        # Count invalid samples
        total_invalid_count += (~valid_mask).sum().item()

        # Collect valid samples
        valid_inputs.append(inputs[valid_mask])
        valid_labels.append(labels[valid_mask])
        valid_mlps.append(mlps[valid_mask])
        valid_one_hots.append(one_hots[valid_mask])

    # Concatenate all valid data
    valid_inputs = torch.cat(valid_inputs, dim=0)
    valid_targets = torch.cat(valid_labels, dim=0)
    valid_mlps = torch.cat(valid_mlps, dim=0)
    valid_one_hots = torch.cat(valid_one_hots, dim=0)

    # Create a new dataset with valid data
    valid_dataset = TensorDataset(valid_inputs, valid_targets)

    # Print total invalid samples
    print(f"\nTotal invalid samples found and removed: {total_invalid_count}")

    return valid_dataset


def count_classes_in_dataloader(dataloader):
    # Initialize a tensor to store class counts (assuming a multi-class classification task)
    class_counts = torch.zeros(15)  # Modify based on your dataset's number of classes

    # Iterate through the DataLoader
    for data in dataloader:
        inputs, one_hots, mlps, labels = data
        # Assuming 'labels' is a tensor of one-hot encoded vectors
        # Sum over the second dimension (class dimension) to count the class occurrences for each instance
        class_counts += labels.sum(dim=0)

    return class_counts

val_dataset = BDB_MLP_Dataset("/Users/abhicmraja/Python Projects/BigDataBowl/Dataset/Test", validate_and_clean=False)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

class_counts = count_classes_in_dataloader(val_dataloader)

# Print the class counts
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count.item()} instances")

# valid_dataset = weed_out_problematic_inputs(val_dataloader)

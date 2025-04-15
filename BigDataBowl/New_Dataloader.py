from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from tqdm import tqdm


class BDB_MLP_Dataset(Dataset):
    def __init__(self, directory, validate_and_clean=False, target_class=12, remove_class_fraction=0.5):
        self.directory = directory
        self.input_dir = os.path.join(directory, 'inputs')
        self.label_dir = os.path.join(directory, 'labels')
        self.persistences_dir = os.path.join(directory, 'persistences')
        self.mlp_dir = os.path.join(directory, 'mlp_inputs')
        self.file_names = [f[:-3] for f in os.listdir(self.input_dir) if f.endswith('.pt')]  # Assuming .pt files
        self.target_class = target_class
        self.remove_class_fraction = remove_class_fraction

        # Optionally validate and remove problematic files
        if validate_and_clean:
            self._validate_and_clean()

        # Remove 50% of class 12 data
        self._remove_class_samples()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        input_path = os.path.join(self.input_dir, file_name + '.pt')
        label_path = os.path.join(self.label_dir, file_name + '.pt')
        persistence_path = os.path.join(self.persistences_dir, file_name + '.pt')
        mlp_path = os.path.join(self.mlp_dir, file_name + '.pt')

        # Load the data
        inputs = torch.load(input_path, weights_only=True).type(torch.float32).permute(2, 0, 1)
        mlps = torch.load(mlp_path, weights_only=True).type(torch.float32)
        persistences = torch.load(persistence_path, weights_only=True).type(torch.float32)[:45]
        labels = torch.load(label_path, weights_only=True).type(torch.float32)

        return inputs, persistences, mlps, labels

    def _validate_and_clean(self):
        """
        Validate all data files in the dataset. If a file contains invalid values (NaN or Inf),
        delete the corresponding files from disk.
        """
        invalid_files = []

        # Use `with` statement to ensure `tqdm` closes properly
        with tqdm(self.file_names, desc="Validating files") as pbar:
            for file_name in pbar:
                input_path = os.path.join(self.input_dir, file_name + '.pt')
                label_path = os.path.join(self.label_dir, file_name + '.pt')
                persistence_path = os.path.join(self.persistences_dir, file_name + '.pt')
                mlp_path = os.path.join(self.mlp_dir, file_name + '.pt')

                try:
                    # Load and validate data
                    inputs = torch.load(input_path, weights_only=True).type(torch.float32).permute(2, 0, 1)
                    mlps = torch.load(mlp_path, weights_only=True).type(torch.float32)
                    persistences = torch.load(persistence_path, weights_only=True).type(torch.float32)[:45]
                    labels = torch.load(label_path, weights_only=True).type(torch.float32)

                    # Check for NaN or Inf values
                    if not torch.isfinite(inputs).all() or \
                            not torch.isfinite(mlps).all() or \
                            not torch.isfinite(persistences).all() or \
                            not torch.isfinite(labels).all():
                        invalid_files.append(file_name)

                except Exception as e:
                    print(f"Error loading file {file_name}: {e}")
                    invalid_files.append(file_name)

        # Delete invalid files
        with tqdm(invalid_files, desc="Deleting invalid files") as pbar:
            for file_name in pbar:
                print(f"Deleting invalid file: {file_name}")
                os.remove(os.path.join(self.input_dir, file_name + '.pt'))
                os.remove(os.path.join(self.label_dir, file_name + '.pt'))
                os.remove(os.path.join(self.persistences_dir, file_name + '.pt'))
                os.remove(os.path.join(self.mlp_dir, file_name + '.pt'))

        print(f"Validation complete. Removed {len(invalid_files)} invalid files.")

    def _remove_class_samples(self):
        all_labels = []
        for file_name in self.file_names:
            label_path = os.path.join(self.label_dir, file_name + '.pt')
            labels = torch.load(label_path, weights_only=True).type(torch.float32)
            labels = labels.view(1, -1)
            all_labels.append(labels)

        print(len(all_labels))
        # Convert to tensor
        all_labels = torch.cat(all_labels, dim=0)
        print(all_labels.shape)

        # Find indices of the target class (class 12)
        target_class_indices = torch.where(all_labels[:, self.target_class] == 1)[0]

        # Ensure target_class_indices isn't empty
        if len(target_class_indices) == 0:
            print(f"No samples found for class {self.target_class}.")
            return

        # Randomly select 50% of the target class indices to remove
        np.random.shuffle(target_class_indices.numpy())
        num_to_remove = len(target_class_indices) * 9 // 10
        indices_to_remove = target_class_indices[:num_to_remove]

        # Convert indices to remove to a set for faster comparison
        indices_to_remove_set = set(indices_to_remove.numpy())

        # Filter the remaining indices by excluding those in indices_to_remove
        remaining_indices = [i for i in range(len(self.file_names)) if i not in indices_to_remove_set]

        # Update the file names to only include the remaining indices
        self.file_names = [self.file_names[i] for i in remaining_indices]

        print(f"Removed {num_to_remove} samples from class {self.target_class}.")


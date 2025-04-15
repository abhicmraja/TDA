from torch.utils.data import Dataset, DataLoader
import os
import torch


class BDB_MLP_Dataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.input_dir = os.path.join(directory, 'inputs')
        self.label_dir = os.path.join(directory, 'labels')
        self.persistences_dir = os.path.join(directory, 'persistences')
        self.mlp_dir = os.path.join(directory, 'mlp_inputs')
        self.file_names = [f[:-3] for f in os.listdir(self.input_dir) if f.endswith('.pt')] #assuming .pt files

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        input_path = os.path.join(self.input_dir, file_name + '.pt')
        label_path = os.path.join(self.label_dir, file_name + '.pt')
        persistence_path = os.path.join(self.persistences_dir, file_name + '.pt')
        mlp_path = os.path.join(self.mlp_dir, file_name + '.pt')

        inputs = torch.load(input_path, weights_only=True)
        input_tensor = inputs.type(torch.float32).permute(2, 0, 1)

        mlps = torch.load(mlp_path, weights_only=True)
        mlp_tensor = mlps.type(torch.float32)

        persistences = torch.load(persistence_path, weights_only=True)

        # Select the top 100 points using the sorted indices
        one_hot_tensor = persistences[:45]
        one_hot_tensor = one_hot_tensor.type(torch.float32)
        labels = torch.load(label_path, weights_only=True)
        label_tensor = labels.type(torch.float32)

        return input_tensor, one_hot_tensor, mlp_tensor, label_tensor


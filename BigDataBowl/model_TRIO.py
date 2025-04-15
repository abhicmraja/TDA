import torch
import torch.nn as nn
import torch.nn.functional as F


class PDE(torch.nn.Module):
    def __init__(self, input_size):
        super(PDE, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Pass through the fully connected layers
        x = x.view(batch_size, -1)
        x = self.model(x)
        # Apply global max pooling across the feature dimension
        # x, _ = torch.max(x, dim=1, keepdim=True)
        return x


class FCNet(torch.nn.Module):
    def __init__(self, MCr, layer_sizes=None):
        M, C, r = MCr
        input_size = M
        if layer_sizes is None:
            layer_sizes = [C//r, ]  # List of layer sizes
        output_size = C
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.LeakyReLU(negative_slope=0.05))
            prev_size = size
        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()

        # Stage 1: Large kernel convolution without padding
        self.stage1 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),  # No padding
            nn.BatchNorm2d(16)
        )

        # Stage 2: Smaller kernel convolutions
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )

        # Stage 3: Additional convolutional stage
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, padding=0),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=2, padding=0),
            nn.BatchNorm2d(128)
        )

        # Final Stage: Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Reduces spatial dimensions to 1x1
        self.PDE = PDE(45 * 4)
        self.fc1 = FCNet([64, 16, 1], [48, 32])
        self.fc2 = FCNet([64, 32, 1], [48, 48])
        self.fc_mlp = FCNet([64, 10, 1], [48, 24])

    def forward(self, x, one_hot):
        # Backbone forward pass
        topo_vec = self.PDE(one_hot)

        x = self.stage1(x)
        topo_resized = self.fc1(topo_vec)
        topo_resized = topo_resized.view(-1, 16, 1, 1)
        topo_resized = topo_resized.expand_as(x)
        x = x * topo_resized

        x = self.stage2(x)
        topo_resized = self.fc2(topo_vec)
        topo_resized = topo_resized.view(-1, 32, 1, 1)
        topo_resized = topo_resized.expand_as(x)
        x = x * topo_resized

        x = self.stage3(x)

        x = self.global_avg_pool(x)  # Output: [batch_size, 128, 1, 1]
        x = torch.flatten(x, 1)  # Flatten to [batch_size, 128]

        pde_output = self.fc_mlp(topo_vec)

        return x, pde_output


class TRIO(nn.Module):
    def __init__(self):
        super(TRIO, self).__init__()
        self.backbone = CNNBackbone()

        # MLP Head
        self.fc1 = nn.Linear(128 + 10 + 8, 64)  # Concatenated inputs
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 15)  # Output size 15 (for classification)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for classification

    def forward(self, cnn_input, one_hot, mlp_input):
        # Pass through the backbone
        backbone_output, pde_output = self.backbone(cnn_input, one_hot)  # Output: [batch_size, 128]

        # Concatenate additional inputs
        combined_input = torch.cat((backbone_output, pde_output, mlp_input), dim=1)  # [batch_size, 128 + 10]

        # MLP Head with LeakyReLU activations
        x = self.activation(self.fc1(combined_input))  # [batch_size, 64]
        x = self.activation(self.fc2(x))  # [batch_size, 32]
        x = self.fc3(x)  # [batch_size, 2]

        # Softmax for classification probabilities
        x = self.softmax(x)  # [batch_size, 2]
        return x


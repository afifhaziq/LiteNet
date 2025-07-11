import torch
import torch.nn as nn

class LiteNet(nn.Module):
    def __init__(self, sequence, features, num_class):
        super(LiteNet, self).__init__()

        self.sequence = sequence  # Store sequence
        self.features = features

        print(f"Initializing LiteNet with sequence={sequence}, features={features}")
        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(sequence, 16, kernel_size=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(sequence, 24, kernel_size=1),
            nn.Conv1d(24, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(sequence, 8, kernel_size=1),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(sequence, 16, kernel_size=1),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(4)
        self.fc1 = nn.Linear(256, 128)
        self.activation5 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.activation6 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x):
        #print(f"Input shape at epoch start: {x.shape}")
        x = x.view(-1, self.sequence, self.features)  # Reshape input

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        conv_out = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)

        pool_out = self.global_pool(conv_out).flatten(start_dim=1)
        #print(x.shape)
        
        fc1_out = self.activation5(self.fc1(pool_out))
        fc2_out = self.activation6(self.fc2(fc1_out))
        x = self.fc3(fc2_out)

        
        return x


class NtCNN(nn.Module):
    def __init__(self, sequence, features, num_class):
        super(NtCNN, self).__init__()

        self.sequence = sequence  # Store sequence
        self.features = features

        print(f"Initializing NtCNN with sequence={sequence}, features={features}")
        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(sequence, 16, kernel_size=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(sequence, 24, kernel_size=1),
            nn.Conv1d(24, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(sequence, 8, kernel_size=1),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(sequence, 16, kernel_size=1),
            nn.ReLU()
        )

        self.global_pool = nn.AvgPool1d(kernel_size=5, stride=5)
        self.fc1 = nn.Linear(256, 128)
        self.activation5 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.activation6 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x, return_activations=False):
        x = x.view(-1, self.sequence, self.features)  # Reshape input

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        conv_out = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)

        pool_out = self.global_pool(conv_out).flatten(start_dim=1)
        
        fc1_out = self.activation5(self.fc1(pool_out))
        fc2_out = self.activation6(self.fc2(fc1_out))
        x = self.fc3(fc2_out)
        

        if return_activations:
            return x, branch1x1, branch3x3, branch5x5, branch_pool, conv_out, pool_out, fc1_out, fc2_out
        return x
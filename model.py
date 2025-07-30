import torch
import torch.nn as nn
import torch.ao.quantization

class QuantizedLiteNet(nn.Module):
    """A wrapper class for INT8 static quantization."""
    def __init__(self, model_fp32):
        super(QuantizedLiteNet, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

class LiteNet(nn.Module):
    def __init__(self, sequence, features, num_class, vocab_size=None, embedding_dim=None):
        super(LiteNet, self).__init__()

        self.sequence = sequence
        self.features = features
        self.embedding = None

        if vocab_size and embedding_dim:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            conv_in_channels = embedding_dim
        else:
            conv_in_channels = sequence

        print(f"Initializing LiteNet with sequence={sequence}, features={features}")

        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(conv_in_channels, 16, kernel_size=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(conv_in_channels, 24, kernel_size=1),
            nn.Conv1d(24, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(conv_in_channels, 8, kernel_size=1),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(conv_in_channels, 16, kernel_size=1),
            nn.ReLU()
        )

        self.global_pool = nn.AvgPool1d(kernel_size=5, stride=5)
        #self.global_pool = nn.AdaptiveAvgPool1d(4)
        self.fc1 = nn.Linear(256, 128)
        self.activation5 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.activation6 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x):
        #print(f"Input shape at epoch start: {x.shape}")

        if self.embedding:
            x = x.squeeze(1)
            x = self.embedding(x.long())  # -> (batch_size, seq_len, embedding_dim)
            # Permute to (batch_size, embedding_dim, seq_len) for Conv1d
            x = x.permute(0, 2, 1)
        else:
            # Original behavior
            x = x.view(-1, self.sequence, self.features)  # Reshape input
 
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        conv_out = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)
        #print("Pool_out shape: ", conv_out.shape)
        pool_out = self.global_pool(conv_out).flatten(start_dim=1)
        #print("pool_out_flat shape: ", pool_out.shape)
        
        fc1_out = self.activation5(self.fc1(pool_out))
        fc2_out = self.activation6(self.fc2(fc1_out))
        x = self.fc3(fc2_out)

        return x


class LiteNetLarge(LiteNet):
    """
    A larger variant of LiteNet that supports variable input sequence length.
    Uses AdaptiveAvgPool1d(4) to allow variable input size and dynamically initializes fc1.
    """
    def __init__(self, sequence, features, num_class):
        super().__init__(sequence, features, num_class)
        print(f"Initializing LiteNetLarge with sequence={sequence}, features={features}")
        # Override global_pool to use adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool1d(4)
        # Defer fc1 initialization
        self.fc1 = None

    def _init_fc1(self, pool_out):
        in_features = pool_out.shape[1] * pool_out.shape[2]
        self.fc1 = nn.Linear(in_features, 128).to(pool_out.device)

    def forward(self, x):
        x = x.view(-1, self.sequence, self.features)
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        conv_out = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)
        pool_out = self.global_pool(conv_out)
    
        pool_out_flat = pool_out.flatten(start_dim=1)
       
        if self.fc1 is None:
            self._init_fc1(pool_out)
        fc1_out = self.activation5(self.fc1(pool_out_flat))
        fc2_out = self.activation6(self.fc2(fc1_out))
        out = self.fc3(fc2_out)
        return out
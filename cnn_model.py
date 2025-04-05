import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
                         ██████ ███    ██ ███    ██ 
                        ██      ████   ██ ████   ██ 
                        ██      ██ ██  ██ ██ ██  ██ 
                        ██      ██  ██ ██ ██  ██ ██ 
                         ██████ ██   ████ ██   ████ 
                                                    
                                                    
      CNN with fixed parameters for any input:
    """
    def __init__(self,
                 convolutions = 5,
                 pooling_dim = 32,
                 embeding_dim = 64,
                 hidden_dim = 64,
                 unembeding_dim = 8,
                 ):
        super().__init__()
        self.convolutions = convolutions
        self.pooling_dim = pooling_dim
        self.embeding_dim = embeding_dim
        self.hidden_dim = hidden_dim
        self.unembeding_dim = unembeding_dim

        self.conv = nn.Conv2d(2, 2, kernel_size=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_compress = nn.Sequential(
          nn.AdaptiveAvgPool2d((pooling_dim, pooling_dim)),  # Force output to (4, 4)
          nn.Flatten(),
          nn.Linear(pooling_dim*pooling_dim*2, embeding_dim)  # Fixed input size (target_dim)
        )
        
        self.fcn = nn.Sequential(
            nn.Linear(embeding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, unembeding_dim)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (1, 2, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (1, 4)
        """
        x = x.float()  # Ensure float32
        original_input = x
        
        # Convolution + Pooling
        for _ in range(self.convolutions):
          x = self.conv(x)       # Shape: (1, 2, H, W) → (1, 2, H, W) (padding=1 preserves size)
          x = self.pool(x)       # Shape: (1, 2, H//2, W//2)
        
        # Spatial compression (flatten + project)
        batch_size, C, H, W = x.shape
        x = self.spatial_compress(x)  # Shape: (1, target_dim)
        
        # FCN
        x = self.fcn(x)              # Shape: (1, 8)
        x = x.view(batch_size, 2, self.unembeding_dim // 2)  # Shape: (1, 2, 4)
        
        # Restore original dimensionality
        original_flat = original_input.view(batch_size, 2, -1).permute(0, 2, 1)
        decoded = torch.bmm(original_flat, x)  # Shape: (1, H*W, 4)
        output = F.softmax(decoded, dim=-1)
        return output.squeeze(0)  # Shape: (H*W, 4)
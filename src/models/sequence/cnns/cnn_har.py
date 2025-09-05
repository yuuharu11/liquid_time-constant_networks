import torch
import torch.nn as nn

class CNN_UCI(nn.Module):
    """
    A simple 1D CNN model for time-series classification.
    """
    def __init__(self, d_model: int, d_output: int):
        """
        Initializes the CNN model.

        Args:
            d_model (int): The number of input features (channels). 
                              For UCI-HAR, this is 9.
            d_output (int): The number of output classes. 
                               For UCI-HAR, this is 6.
        """
        super(CNN_UCI, self).__init__()
        
        # 畳み込みブロック1
        self.conv1 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 畳み込みブロック2
        self.conv2 = nn.Conv1d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 全結合層
        # プーリング層を2回通過後のシーケンス長を計算する必要があります。
        # UCI-HARのシーケンス長は128 -> pool1 -> 64 -> pool2 -> 32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32, 128) # 128 (out_channels) * 32 (sequence_length)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, d_output)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x, None
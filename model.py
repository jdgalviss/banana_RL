import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        "*** YOUR CODE HERE ***"
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    
class VisualQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv1_units = 128, conv2_units = 128*2, conv3_units = 128*2, fc1_units=2304, fc2_units=1024):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        "*** YOUR CODE HERE ***"
        super(VisualQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        s = 3
        self.conv1 = nn.Conv3d(3, conv1_units, kernel_size=(1, 3, 3), stride=(1,s,s))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,2,2))
        self.bn1 = nn.BatchNorm3d(conv1_units)
        self.conv2 = nn.Conv3d(conv1_units, conv2_units, kernel_size=(1, 3, 3), stride=(1,s,s))
        self.bn2 = nn.BatchNorm3d(conv2_units)
        self.conv3 = nn.Conv3d(conv2_units, conv3_units, kernel_size=(4, 3, 3), stride=(1,s,s))
        self.bn3 = nn.BatchNorm3d(conv3_units)
        self.fc1 = nn.Linear(fc1_units, fc2_units)
        self.fc2 = nn.Linear(fc2_units, action_size)
        
    def _cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = torch.squeeze(x,0)
        #print(x.size())
        x = self._cnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
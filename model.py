import torch
import torch.nn as nn

class DinoDetector(nn.Module):
    def __init__(self, input_size):
        super(DinoDetector, self).__init__()
        self.shared_fc = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.class_fc = nn.Linear(256, 2)
        self.bbox_fc = nn.Linear(256, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.shared_fc(x))
        x = self.dropout(x)
        class_output = self.class_fc(x)
        bbox_output = self.sigmoid(self.bbox_fc(x))
        return class_output, bbox_output
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, dataset_train_tensor):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(dataset_train_tensor.shape[1], 2)  # Asumiendo 2 clases

    def forward(self, x):
        return self.linear(x)
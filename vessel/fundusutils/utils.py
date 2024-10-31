from torch import nn

def initialize_weights(m):
    """
    Initialize the weights of the model layers.

    Args:
    m (nn.Module): A module in the neural network

    Note:
    - Uses Kaiming initialization for Conv2d and Linear layers
    - For BatchNorm2d, weight is set to 1 and bias to 0
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
        



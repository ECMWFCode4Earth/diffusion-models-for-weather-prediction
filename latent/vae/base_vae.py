from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    def __init__(self,) -> None:
        super(BaseVAE, self).__init__()

    def encode(self,):
        raise NotImplementedError
    
    def decode(self,):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def generate(self):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass
        
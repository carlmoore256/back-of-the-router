
from dataset import create_attribute_dict

# an object that contains a generated botr
class BOTR():
  
  def __init__(self, config):
    self.layers = []
    self.config = config
    self.attributes = create_attribute_dict()

  
  def append_layer(self, layer):
    self.layers.append(layer)

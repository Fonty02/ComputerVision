from .oxford_pets import OxfordPets
from .oxford_flowers import OxfordFlowers
#from .fgvc_aircraft import FGVCAircraft
from .dtd import DescribableTextures
from .eurosat import EuroSAT
from .food101 import Food101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .ucf101 import UCF101
from .imagenet import ImageNet
from .artgraph import Artgraph

__all__ = [
    'OxfordPets',
    'OxfordFlowers', 
    'FGVCAircraft', 
    'DescribableTextures', 
    'EuroSAT', 
    'StanfordCars',
    'Food101',
    'SUN397',
    'Caltech101',
    'UCF101',
    'ImageNet',
    'ArtGraph'
]

def build_dataset(dataset_name, *args, **kwargs):
    # Add new datasets here as needed
    if dataset_name == 'oxford_pets':
        return OxfordPets(*args, **kwargs)
    elif dataset_name == 'oxford_flowers':
        return OxfordFlowers(*args, **kwargs)
    #elif dataset_name == 'fgvc_aircraft':
        #return FGVCAircraft(*args, **kwargs)
    elif dataset_name == 'dtd':
        return DescribableTextures(*args, **kwargs)
    elif dataset_name == 'eurosat':
        return EuroSAT(*args, **kwargs)
   # elif dataset_name == 'stanford_cars':
    #    return StanfordCars(*args, **kwargs)
    elif dataset_name == 'food101':
        return Food101(*args, **kwargs)
    elif dataset_name == 'sun397':
        return SUN397(*args, **kwargs)
    elif dataset_name == 'caltech101':
        return Caltech101(*args, **kwargs)
    elif dataset_name == 'ucf101':
        return UCF101(*args, **kwargs)
  #  elif dataset_name == 'artgraph_style':
       # return ArtgraphStyle(*args, **kwargs)
    elif dataset_name == 'artgraph':
        return Artgraph(*args, **kwargs)
    else:
        raise ValueError('Dataset not found: {}'.format(dataset_name))
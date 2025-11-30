"""
MILK10k Skin Lesion Classification Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import config
from . import utils
from . import data_preprocessing
from . import dataset
from . import models
from . import evaluate

__all__ = [
    'config',
    'utils',
    'data_preprocessing',
    'dataset',
    'models',
    'evaluate'
]

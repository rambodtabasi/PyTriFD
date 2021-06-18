"""Top-level package for PyTrilinos Finite Difference Code."""

__author__ = """John T. Foster"""
__email__ = 'johntfosterjr@gmail.com'
__version__ = '0.1.0'

from .PyTriFD import FD
from .ensight import Ensight

__all__ = ['FD', 'Ensight']

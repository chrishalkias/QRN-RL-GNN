# -*- coding: utf-8 -*-
# src/__init__.py

'''
Created Wed 24 Apr 2025
The src initialization to be treated as a package.
'''
import logging
from pathlib import Path

from repeaters import RepeaterNetwork
from models import CNN, GNN
from src.cnn_env import Environment

# core metadata
__version__ = "1.0"
__author__ = "Chris Chalkias"
__email__ = "chalkias@lorentz.leidenuniv.nl"
__license__ = "https://hdl.handle.net/1887/license:7"
__url__ = "https://github.com/chrishalkias/QRN-RL-GNN/src"
__dependencies__ = []
__all__ = ['repeaters', 'models', 'environment', 'main']

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Loaded {__name__} v{__version__}")

# Runtime paths
DATA_DIR = Path(__file__).parent / "data"
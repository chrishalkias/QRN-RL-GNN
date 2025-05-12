# -*- coding: utf-8 -*-
# src/trial_and error.py

'''
Created Fri 09 May 2025
Test things here
'''

from gnn_env import Environment
from models import GNN

n=4
[]
lista = [f'self.swapAT({i})' if (i != 0) and (i !=n-1) else '' for i in range(n)]
print(lista if lista else None)
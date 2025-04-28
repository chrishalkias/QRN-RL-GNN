# -*- coding: utf-8 -*-
# game/play.py

'''
Created Wed 24 Apr 2025
The configuration and run file.
'''

from game import Qonnect

n = 6
game_config = {
    'grid_size': n,                      # grid
    'cell_size': 100,                    # Smaller cells
    'dot_color': (150, 50, 200),         # Purple dots
    'merge_enabled': True,               # Enable merging
    'window_title': "Qonnect",           # Title of the window
    'diagonal_color': (255, 0, 0),       # Red diagonal line
    'text_padding': 20,                  # Space for row/column numbers
    'dot_lifetime': 10,                  # Number of actions before dot disappears
    'title_font_size': 36,               # Size for the Qonnect title
    'selection_color': (100, 200, 255),  # Brighter selection color
    'selection_thickness': 6,            # Thicker selection border
    'dot_creation_prob': 1,              # chance to create a dot when placing
    'dot_merge_prob': 1,                 # Probability of merging dots
}
game = Qonnect(game_config)

if __name__ == "__main__":
    game.run()
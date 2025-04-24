# -*- coding: utf-8 -*-
# game/game.py

'''
Created Wed 15 Apr 2025
A simple quantum network based board game.
'''

import pygame
import random
import sys
from typing import List, Tuple, Optional, Dict

class Qonnect:
    def __init__(self, config: dict):
        """Initialize the game with customizable parameters"""
                                                
        self.config = {                         # Game configuration
            'grid_size': 5,                     # Size of the grid (N x N)
            'cell_size': 80,                    # Size of each cell in pixels
            'padding': 1,                       # Padding between cells
            'bg_color': (240, 240, 240),        # Background color
            'grid_color': (200, 200, 200),      # Grid lines color
            'dot_color': (255, 100, 100),       # Dot color
            'target_position': (0, 0),          # Will be set to top-right corner
            'target_color': (100, 255, 100),    # Target cell color
            'merge_enabled': True,              # Whether merging is allowed
            'window_title': "Qonnect",          # Title of the window
            'black_cells': [],                  # Cells that are always black
            'grey_cells': [],                   # Cells where dots can be placed
            'diagonal_color': (255, 0, 0),      # Color for the diagonal line
            'text_color': (0, 0, 0),            # Color for row/column numbers
            'text_padding': 20,                 # Padding for row/column numbers
            'dot_lifetime': 5,                  # Number of actions before dot disappears
            'title_font_size': 48,              # Size for the Qonnect title
            'selection_color': (100, 200, 255), # Color for selected cells
            'selection_thickness': 5,           # Thickness of selection border
            'dot_creation_prob': 0.8,           # Probability of creating a dot when placing
            'dot_merge_prob': 0.8,              # Probability of merging dots
        }

        self.action_log = []
        
        # Update with provided config
        self.config.update(config)
        
        # Set target position to top-right corner (and bottom-left by symmetry)
        self.config['target_position'] = (0, self.config['grid_size']-1)
        
        # Initialize pygame
        pygame.init()
        
        # Calculate window size (with extra space for numbers and title)
        window_size = (
            self.config['grid_size'] * (self.config['cell_size'] + self.config['padding']) + 
            self.config['padding'] + self.config['text_padding'] + 200,  # Added 200 for panel
            self.config['grid_size'] * (self.config['cell_size'] + self.config['padding']) + 
            self.config['padding'] + self.config['text_padding'] + self.config['title_font_size']
        )
        
        # Create window
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(self.config['window_title'])
        
        # Initialize special cells
        self.initialize_special_cells()
        
        # Game state
        self.grid = [[False for _ in range(self.config['grid_size'])] 
                    for _ in range(self.config['grid_size'])]
        self.dot_timers: Dict[Tuple[int, int], int] = {}  # Tracks dot lifetimes
        self.selected_cells = []
        self.game_over = False
        self.win = False
        self.action_count = 0  # Counts player actions for decay


    def show_rules_popup(self):
        """Display game rules in a modal popup"""
        rules_lines = [
            "Welcome to Qonnect! ",
            "The goal of the game is to put a dot on the green square (end-to-end entanglement). ",
            "Here are the rules: ",
            "Rule 0: The game is played on a grid of size N x N with reflection symmetry about the diagonal. ",
            "Rule 1: You can select a grey cell to place a dot (entanglement generation). ",
            "Rule 2: You can select two dots to merge them (entanglement swapping). ",
            "When swapping, the dots must be in the same row or column and selection must be first done on the topmost or leftmost dot. ",
            "For example, if you have dots at (i,j) and (j,k), you can swap them. ",
            "You can also swap dots with their mirror images and sometimes only with their mirror images. ",
            "The mirror image of a dot (i,j) is (j,i). ",
            "If you swap two dots (i,j) and (j,k), the result will be in the position (i,k). ",
            "Rule 3: The dots will disappear after a few actions. ",
            "Rule 4: You can only place dots in grey cells. ",
            "Rule 5: The game ends when you create a dot in the target position. ",
            "Final rule: Enjoy the game!               -Chris"
        ]
        rules_text = "".join(rules_lines)
        
        # Create a semi-transparent overlay
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Create popup rectangle
        popup_width, popup_height = 700, 500
        popup_x = (self.screen.get_width() - popup_width) // 2
        popup_y = (self.screen.get_height() - popup_height) // 2
        
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (popup_x, popup_y, popup_width, popup_height))
        
        # Render rules text (with word wrapping)
        font = pygame.font.SysFont(None, 24)
        y_offset = popup_y + 20
        for line in self._wrap_text(rules_text, font, popup_width - 40):
            text = font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (popup_x + 20, y_offset))
            y_offset += 30
        
        # Render close button
        close_rect = pygame.Rect(popup_x + popup_width - 120, popup_y + popup_height - 50, 100, 30)
        pygame.draw.rect(self.screen, (200, 0, 0), close_rect)
        close_text = font.render("Play", True, (255, 255, 255))
        self.screen.blit(close_text, (close_rect.x + 30, close_rect.y + 5))
        
        pygame.display.flip()
        
        # Wait for user to click close
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if close_rect.collidepoint(event.pos):
                        waiting = False

    def _wrap_text(self, text, font, max_width):
        """Helper to wrap text to fit within max_width"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def initialize_special_cells(self):
        """Initialize the black and grey cells based on grid size"""
        size = self.config['grid_size']
        self.config['black_cells'] = [(i, i) for i in range(size)]  # Main diagonal
        self.config['grey_cells'] = []
        
        # Secondary diagonal (i, i+1) and (i-1, i)
        for i in range(size):
            if i + 1 < size:
                self.config['grey_cells'].append((i, i + 1))
            if i - 1 >= 0:
                self.config['grey_cells'].append((i, i - 1))
        
        # Remove duplicates and ensure they're unique
        self.config['grey_cells'] = list(set(self.config['grey_cells']))
        # Remove any grey cells that are also black cells
        self.config['grey_cells'] = [cell for cell in self.config['grey_cells'] if cell not in self.config['black_cells']]
    
    def pos_to_grid(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert screen position to grid coordinates"""
        x, y = pos
        text_padding = self.config['text_padding']
        title_height = self.config['title_font_size']
        cell_size = self.config['cell_size'] + self.config['padding']
        
        # Check if position is within grid bounds (excluding number and title areas)
        if (x < text_padding or y < title_height or
            x >= text_padding + self.config['grid_size'] * cell_size or 
            y >= title_height + self.config['grid_size'] * cell_size):
            return None
            
        col = (x - text_padding) // cell_size
        row = (y - title_height) // cell_size
        return (row, col)
    
    def draw_grid(self):
        """Draw the grid and dots"""
        self.screen.fill(self.config['bg_color'])
        cell_size = self.config['cell_size']
        padding = self.config['padding']
        text_padding = self.config['text_padding']
        title_height = self.config['title_font_size']
        
        # Draw game title "Qonnect"
        title_font = pygame.font.SysFont(None, self.config['title_font_size'])
        title_text = title_font.render("Qonnect", True, (0, 0, 0))
        title_rect = title_text.get_rect(center=(self.screen.get_width()//2, title_height//2))
        self.screen.blit(title_text, title_rect)
        
        # Draw row and column numbers
        font = pygame.font.SysFont(None, 24)
        for i in range(self.config['grid_size']):
            # Column numbers (top)
            text = font.render(str(i+1), True, self.config['text_color'])
            text_rect = text.get_rect(
                center=(text_padding + i * (cell_size + padding) + cell_size // 2, 
                        title_height + text_padding // 2)
            )
            self.screen.blit(text, text_rect)
            
            # Row numbers (left side)
            text = font.render(str(i+1), True, self.config['text_color'])
            text_rect = text.get_rect(
                center=(text_padding // 2,
                        title_height + text_padding + i * (cell_size + padding) + cell_size // 2)
            )
            self.screen.blit(text, text_rect)
        
        # Draw grid lines
        for i in range(self.config['grid_size'] + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen, self.config['grid_color'],
                (text_padding, title_height + text_padding + i * (cell_size + padding)),
                (text_padding + self.config['grid_size'] * (cell_size + padding), 
                title_height + text_padding + i * (cell_size + padding)),
                2
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, self.config['grid_color'],
                (text_padding + i * (cell_size + padding), title_height + text_padding),
                (text_padding + i * (cell_size + padding), 
                title_height + text_padding + self.config['grid_size'] * (cell_size + padding)),
                2
            )
        
        # Draw black cells (main diagonal)
        for row, col in self.config['black_cells']:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                (
                    text_padding + col * (cell_size + padding),
                    title_height + text_padding + row * (cell_size + padding),
                    cell_size, cell_size
                )
            )
        
        # Draw grey cells (secondary diagonals)
        for row, col in self.config['grey_cells']:
            pygame.draw.rect(
                self.screen, (200, 200, 200),
                (
                    text_padding + col * (cell_size + padding),
                    title_height + text_padding + row * (cell_size + padding),
                    cell_size, cell_size
                )
            )
        
        # Draw red diagonal line (not going over the last square)
        pygame.draw.line(
            self.screen, self.config['diagonal_color'],
            (text_padding, title_height + text_padding),
            (
                text_padding + (self.config['grid_size']) * (cell_size + padding),
                title_height + text_padding + (self.config['grid_size']) * (cell_size + padding)
            ),
            3
        )
        
        # Highlight target positions (top-right and bottom-left)
        target_row, target_col = self.config['target_position']
        pygame.draw.rect(
            self.screen, self.config['target_color'],
            (
                text_padding + target_col * (cell_size + padding),
                title_height + text_padding + target_row * (cell_size + padding),
                cell_size, cell_size
            )
        )
        # Mirror target position
        pygame.draw.rect(
            self.screen, self.config['target_color'],
            (
                text_padding + target_row * (cell_size + padding),
                title_height + text_padding + target_col * (cell_size + padding),
                cell_size, cell_size
            )
        )
        
        # Draw dots with transparency based on age
        dot_radius = cell_size // 4
        for row in range(self.config['grid_size']):
            for col in range(self.config['grid_size']):
                if (row, col) not in self.config['black_cells'] and self.grid[row][col]:
                    # Calculate alpha based on remaining lifetime
                    remaining_life = self.dot_timers.get((row, col), self.config['dot_lifetime'])
                    alpha = max(0, 255 * remaining_life // self.config['dot_lifetime'])
                    
                    # Create a surface with per-pixel alpha
                    dot_surface = pygame.Surface((dot_radius*2, dot_radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(
                        dot_surface, 
                        (*self.config['dot_color'][:3], alpha),  # Add alpha to color
                        (dot_radius, dot_radius), 
                        dot_radius
                    )
                    
                    # Blit the dot surface
                    self.screen.blit(
                        dot_surface,
                        (
                            text_padding + col * (cell_size + padding) + cell_size // 2 - dot_radius,
                            title_height + text_padding + row * (cell_size + padding) + cell_size // 2 - dot_radius
                        )
                    )
        
        # Highlight selected cells (now any cell can be selected)
        for row, col in self.selected_cells:
            pygame.draw.rect(
                self.screen, self.config['selection_color'],
                (
                    text_padding + col * (cell_size + padding),
                    title_height + text_padding + row * (cell_size + padding),
                    cell_size, cell_size
                ),
                self.config['selection_thickness']
            )
        
        # Display game over message
        if self.game_over:
            font = pygame.font.SysFont(None, 50)
            if self.win:
                text = font.render("End-to-end Linked!", True, (0, 100, 0))
            else:
                text = font.render("Game Over", True, (150, 0, 0))
            text_rect = text.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2))
            self.screen.blit(text, text_rect)
        self.draw_side_panel() # Draw side panel with game info
        pygame.display.flip()
    
    def is_valid_placement_position(self, row: int, col: int) -> bool:
        """Check if a position is valid for placing dots (grey cells only)"""
        return (row, col) in self.config['grey_cells']
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is valid (not black)"""
        return (row, col) not in self.config['black_cells']
    
    def place_dot_with_symmetry(self, row: int, col: int):
        """Place a dot at the specified position and its mirror image with probability"""
        if self.is_valid_placement_position(row, col) and not self.grid[row][col]:
            self.age_dots()
            # Probabilistic dot creation
            if random.random() < self.config['dot_creation_prob']:
                self.grid[row][col] = True
                self.dot_timers[(row, col)] = self.config['dot_lifetime']
                
                # Place mirror dot
                mirror_row, mirror_col = col, row
                if mirror_row != mirror_col:  # Don't mirror on diagonal (those are black anyway)
                    self.grid[mirror_row][mirror_col] = True
                    self.dot_timers[(mirror_row, mirror_col)] = self.config['dot_lifetime']
                
                # Age all existing dots
                self.log_action(f"Entangle ({row+1},{col+1})")
                self.check_win_condition()
                return True
        return False
    
    def can_merge(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two dots can be merged (not mirrors of each other)"""
        row1, col1 = pos1
        row2, col2 = pos2
        return not (row1 == col2 and col1 == row2)  # Not mirror positions
    
    def merge_dots_with_symmetry(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        """Merge two dots if they exist according to matrix multiplication rules"""
        row1, col1 = pos1
        row2, col2 = pos2
        
        # Check if these are mirror positions (can't merge)
        if not self.can_merge(pos1, pos2):
            return
        
        # Check if both positions have dots
        if self.grid[row1][col1] and self.grid[row2][col2]:
            self.age_dots()
            if random.random() < self.config['dot_merge_prob']:
                # Apply matrix multiplication rule: (i,j) and (k,l) can merge if j == k
                if col1 == row2:
                    # Result goes to (i,l)
                    new_row, new_col = row1, col2
                    
                    # Remove original dots and their mirrors
                    self.remove_dot(row1, col1)
                    self.remove_dot(row2, col2)
                    
                    # Check if target position is valid
                    if self.is_valid_position(new_row, new_col):
                        # Place new dot and its mirror
                        self.grid[new_row][new_col] = True
                        self.dot_timers[(new_row, new_col)] = self.config['dot_lifetime']
                        
                        mirror_row, mirror_col = new_col, new_row
                        if mirror_row != mirror_col:  # Don't mirror on diagonal
                            self.grid[mirror_row][mirror_col] = True
                            self.dot_timers[(mirror_row, mirror_col)] = self.config['dot_lifetime']
                        
                        self.log_action(f"Swap ({row1+1},{col1+1}) and ({row2+1},{col2+1})")

                        # Age all existing dots (including the new ones)
                        
                        # Check win condition
                        self.check_win_condition()
    
    def remove_dot(self, row: int, col: int):
        """Remove a dot and its mirror image"""
        self.grid[row][col] = False
        if (row, col) in self.dot_timers:
            del self.dot_timers[(row, col)]
        
        # Remove mirror dot
        mirror_row, mirror_col = col, row
        if mirror_row != mirror_col:  # Don't mirror on diagonal
            self.grid[mirror_row][mirror_col] = False
            if (mirror_row, mirror_col) in self.dot_timers:
                del self.dot_timers[(mirror_row, mirror_col)]
    
    def age_dots(self):
        """Age all dots by one action"""
        self.action_count += 1
        dots_to_remove = []
        
        # Update timers and mark expired dots
        for pos in list(self.dot_timers.keys()):
            self.dot_timers[pos] -= 1
            if self.dot_timers[pos] <= 0:
                dots_to_remove.append(pos)
        
        # Remove expired dots
        for row, col in dots_to_remove:
            self.remove_dot(row, col)
    
    def check_win_condition(self):
        """Check if the game is won (dot is present in target position)"""
        target_row, target_col = self.config['target_position']
        if self.grid[target_row][target_col] or self.grid[target_col][target_row]:
            self.game_over = True
            self.win = True
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click events"""
        if self.game_over:
            return
        
        grid_pos = self.pos_to_grid(pos)
        if not grid_pos:
            return
        
        row, col = grid_pos
        
        # Don't allow interaction with black cells
        if (row, col) in self.config['black_cells']:
            return
        
        if not self.config['merge_enabled']:
            self.place_dot_with_symmetry(row, col)
        else:
            # In merge mode, first try to place a dot if cell is empty and valid
            if self.is_valid_placement_position(row, col) and not self.grid[row][col]:
                if self.place_dot_with_symmetry(row, col):
                    return  # Successfully placed a dot
            
            # If cell has a dot or couldn't place one, proceed with selection
            # Now any non-black cell can be selected for merging
            if (row, col) in self.selected_cells:
                self.selected_cells.remove((row, col))
            elif self.is_valid_position(row, col):
                # Check if this selection would pair with a mirror of an already selected dot
                if len(self.selected_cells) == 1:
                    selected_row, selected_col = self.selected_cells[0]
                    if (row == selected_col and col == selected_row):
                        return  # Don't allow selecting mirror positions
                
                self.selected_cells.append((row, col))
                
                if len(self.selected_cells) == 2:
                    self.merge_dots_with_symmetry(*self.selected_cells)
                    self.selected_cells = []

    def draw_side_panel(self):
        """Draw the side panel with game information and action log"""
        panel_width = 200
        panel_x = self.screen.get_width() - panel_width
        
        # Draw panel background
        pygame.draw.rect(
            self.screen, (220, 220, 220),
            (panel_x, 0, panel_width, self.screen.get_height())
        )
        
        # Draw title
        font = pygame.font.SysFont(None, 24)
        title = font.render("Game Settings:", True, (0, 0, 0))
        self.screen.blit(title, (panel_x + 10, 10))
        
        # Draw current parameters
        params = [
            f"Creation Prob: {self.config['dot_creation_prob']}",
            f"Swap Prob: {self.config['dot_merge_prob']}",
            f"Link Lifetime: {self.config['dot_lifetime']}",
            f"Actions: {self.action_count}"
        ]
        
        for i, param in enumerate(params):
            text = font.render(param, True, (0, 0, 0))
            self.screen.blit(text, (panel_x + 10, 50 + i * 30))
        
        # Draw action log title
        log_title = font.render("Action Log:", True, (0, 0, 0))
        self.screen.blit(log_title, (panel_x + 10, 180))
        
        # Draw last 10 actions (you'll need to maintain this list)
        if hasattr(self, 'action_log'):
            for i, action in enumerate(self.action_log[-2*self.config['grid_size']:]):  # Show last 10 actions
                text = font.render(action, True, (0, 0, 0))
                self.screen.blit(text, (panel_x + 10, 200 + i * 20))

    def log_action(self, action: str):
        """Add an action to the log"""
        if not hasattr(self, 'action_log'):
            self.action_log = []
        self.action_log.append(action)
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        self.show_rules_popup()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.__init__(self.config)
            
            self.draw_grid()
            clock.tick(60)
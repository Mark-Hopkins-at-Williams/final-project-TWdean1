import pygame as pg
from pygame.locals import *

def getFont(size):
    return pg.font.SysFont("Arial", size)
BLACK = (0,0,0)

class Button:
    def __init__(self,x, y, width, height, color, text= None, text_color = BLACK):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.text_color = BLACK
    
    def draw(self, window):
        pg.draw.rect(window, BLACK, (self.x, self.y, self.width, self.height), 2)
        if self.text:
            buttonFont = getFont(22)
            textSurface = buttonFont.render(self.text, 1, self.text_color)
            window.blit(textSurface, (self.x + self.width /
                                    2 - textSurface.get_width()/2, self.y + self.height/2 - textSurface.get_height()/2))
   
    def clicked(self, pos):
        x, y = pos

        if not (x >= self.x and x <= self.x + self.width):
            return False
        if not (y >= self.y and y <= self.y + self.height):
            return False

        return True


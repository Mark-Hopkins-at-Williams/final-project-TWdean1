import sys, math, torch
import pygame as pg
from pygame.locals import *
from button import *

from frankenstein import build_frankenstein, build_cnn_only, Classifier
from dataManager import DataPartition

WHITE = (255, 255, 255)
BLACK = (0,0,0)
GREEN = (0,0, 255)
DRAWINGCOLOR = BLACK
BGCOLOR = WHITE
gridCode = {WHITE: 0, BLACK: 1}

FPS = 100

WIDTH, HEIGHT = 600, 700
ROWS = COLS = 64
PIXELSIZE = WIDTH//COLS
TOOLBAR = HEIGHT-WIDTH

BUTTONY = HEIGHT- TOOLBAR/2 - 25
buttons = [
    Button(5, BUTTONY, 165, 70, WHITE, "Switch (to CNN)", BLACK), #default model is LSTM 
    Button(175, BUTTONY, 130, 70, WHITE, "Clear", BLACK),
    Button(310, BUTTONY, 130, 70, WHITE, "Classify", BLACK),
    Button (445, BUTTONY, 155, 70, WHITE) #draw the shape we think is on screen in this little box
]

class paint:
    def __init__(self):
        pg.init()
        #state stuff
        self.QUIT = False
        self.DRAWING = False
        self.clock = pg.time.Clock()
        self.grid = self.gridinit(ROWS, COLS)
        self.gridtensor = torch.zeros(8, ROWS, COLS) # modified 
        self.current_stroke = 0 

        #screen stuff
        self.window = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("ShapeShift")

        #cursor stuff
        self.cursorImage = pg.image.load('pencil.png')
        self.cursorRect = self.cursorImage.get_rect()
        pg.mouse.set_visible(False)

        #LSTM model stuff
        test_partition = DataPartition('medium_test_data_json.json', 'test')
        net = build_frankenstein(output_classes = len(test_partition.possible_responses()), dense_hidden_size = 256, LSTM_hidden_size=256, channels_list = [32, 16, 8]) # may need to edit 
        self.LSTM = Classifier(net, test_partition.possible_responses(), None, None, None)
        self.LSTM.load('medium_model_lstm')

        #CNN model stuff
        net = build_cnn_only(output_classes = len(test_partition.possible_responses()), dense_hidden_size = 256, channels_list = [32, 16, 8])
        self.CNN = Classifier(net, test_partition.possible_responses(), None, None, None)
        self.CNN.load("medium_model_cnn")

        self.current_model = self.LSTM 
        self.is_LSTM = True


    def gridinit(self, rows, cols):
        grid = []
        for row in range(ROWS):
            grid.append([])
            for col in range(COLS):
                grid[row].append(BGCOLOR) #each element is RGB length 3 list, allows for more colors in future
        return grid


    def drawgrid(self):
        for i,row in enumerate(self.grid):
            for j, pixel in enumerate(row):
                pg.draw.rect(self.window, pixel,(j *PIXELSIZE, i *PIXELSIZE, PIXELSIZE, PIXELSIZE))

    def draw(self):
        self.window.fill(BGCOLOR)
        self.drawgrid()
        for button in buttons:
            button.draw(self.window)
        self.cursorRect.center = pg.mouse.get_pos()
        self.window.blit(self.cursorImage, self.cursorRect)
        pg.display.update()

    def getrowcol(self, pos):
        x,y = pos
        row = y//PIXELSIZE
        col = x//PIXELSIZE

        if row > ROWS:
            raise IndexError
        
        return row,col

    def classify(self):
        result_dict = self.current_model(self.gridtensor)
        best_result = None
        best_percent = -float('inf')
        for result in result_dict:
            if result_dict[result] > best_percent:
                best_result = result
                best_percent = result_dict[result]
        return f"{best_result}, ({round(best_percent, 2)})"

    def run(self):
        while not self.QUIT:
            self.draw()
            self.clock.tick(FPS)
            self.events = pg.event.get()
            
            if self.current_stroke == 8: 
                self.grid = self.gridinit(ROWS, COLS)
                self.gridtensor = torch.zeros(8, ROWS, COLS)

            for event in self.events:

                #can quit game using red "x" in corner or escape key
                if event.type == pg.QUIT:
                    self.QUIT = True
                    buttons[-1].text = ""
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.QUIT = True
                        buttons[-1].text = ""

                elif event.type == pg.MOUSEBUTTONDOWN:
                    pos = event.pos[0], event.pos[1]

                    try: 
                        self.DRAWING = True
                        row,col = self.getrowcol(pos)
                        self.grid[row][col] = DRAWINGCOLOR
                        self.gridtensor[self.current_stroke, row, col] = 1
                    except IndexError: #user clicks in the toolbar
                        self.DRAWING = False
                        for button in buttons: 
                            if not button.clicked(pos):
                                continue
                            elif button.text == "Clear":
                                self.grid = self.gridinit(ROWS, COLS)
                                self.gridtensor = torch.zeros(8, ROWS, COLS) # modified 
                                buttons[-1].text = " "
                                self.current_stroke = 0 
                            elif "Switch" in button.text:
                                if self.is_LSTM:
                                    self.current_model = self.CNN
                                    button.text = "Switch (to LSTM)"
                                    self.is_LSTM = False
                                else:
                                    self.current_model = self.LSTM
                                    button.text = "Switch (to CNN)"
                                    self.is_LSTM = True
                            elif button.text == "Classify":
                                result = self.classify()
                                buttons[-1].text = result
                elif event.type == pg.MOUSEMOTION and self.DRAWING:
                    pos = event.pos[0], event.pos[1]
                    try: 
                        row,col = self.getrowcol(pos)
                        self.grid[row][col] = DRAWINGCOLOR
                        self.gridtensor[self.current_stroke, row, col] = 1
                    except IndexError: #user clicks outside of the drawable screen
                        pass
                elif event.type == pg.MOUSEBUTTONUP and self.DRAWING:
                    self.DRAWING = False
                    self.current_stroke +=1 
                    if self.current_stroke < 9:
                        temp_tensor = self.gridtensor[self.current_stroke-1].detach()
                        self.gridtensor[self.current_stroke] = temp_tensor


            pg.display.flip()

        pg.quit()

if __name__ == "__main__":
    mypaint = paint()
    mypaint.run()

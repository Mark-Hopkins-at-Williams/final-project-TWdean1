from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
import os

import json

from utility_methods import list_files_in_folder

__all__ = ["read_ndjson_file", "convertStrokeVectorToBitmap"]

def read_ndjson_file(listOfPathToRawData, data_output_folder, json_output_name, trunc_len = None, three_d = False, compress = False, output_image_size = 256, pad_len = 10):
    """
    Given a path to JSON formatted training images in stroke form and a folder to output to,
    converts each image to a bitmap in JSON format and places it into its own file in 
    output folder. Also creates a dataLoader data.json file in Mark's style.

    If thre_d = True, then creates three dimensional tensors where each subsequent stroke is added to 
    the next frame of the tensor. This tensor can have dimension 0 of pad_len. 
    """
    # create output folder if it does not already exist: 
    os.makedirs(data_output_folder, exist_ok=True)
    # Counter for output file naming
    # Data for dataLoader JSON file
    dataLoaderData = []

    # Open raw data file
    for path in listOfPathToRawData:
        
        counter = 0
        with open(path, 'r') as rawDataFile:

            for line in tqdm(rawDataFile, desc = f"Processing: {path}"):
                # Parse the JSON data from each line. Each line is an image
                record = json.loads(line)
                
                # Extract the 'word' (shape) as a string and 'drawing' (stroke arrays) as a nested list
                shape = str(record.get('word'))
                drawing = record.get('drawing')

                # remove instances where strokes > pad_len
                if len(drawing) > pad_len:
                    continue

                # Write one image to a new named output file
                singleImagePath = f"{data_output_folder}/{shape}{str(counter)}.json"
                singleImagePath = "".join(singleImagePath.split()) # remove whitespace from filepath
                with open(singleImagePath, 'w') as newOutputFile:
                    # make 3d tensor 
                    if three_d:
                        sequential_strokes = []
                        three_d_image_data = []
                        for x_y_cords in drawing:
                            sequential_strokes.append(x_y_cords) # add one stroke at a time
                            image = convertStrokeVectorToBitmap(sequential_strokes, size=output_image_size)
                            if compress:
                                # make 64 x 64 image
                                image_tensor = torch.tensor(image).unsqueeze(dim=0)
                                compressor = nn.MaxPool2d(kernel_size = 4, stride = 4)
                                compressed_image = compressor(image_tensor).squeeze(dim=0)
                                image = compressed_image.tolist()

                            three_d_image_data.append(image) # generate image and add to list
                        
                        # pad the image  
                        while len(three_d_image_data) < pad_len:
                            pad_image = None
                            if compress:
                                pad_image = torch.zeros(64, 64).tolist()
                            else:
                                pad_image = torch.zeros(256, 256).tolist()
                            three_d_image_data.append(pad_image)

                        # add the 3d image data to file 
                        json.dump({'image' : three_d_image_data}, newOutputFile)

                    # 2d tensor with all strokes 
                    else:
                    # Make and add the bitmap
                        single_image_tensor = torch.tensor(convertStrokeVectorToBitmap(drawing, size=output_image_size)).unsqueeze(dim=0)
                        compressor = nn.MaxPool2d(kernel_size = 4, stride = 4)
                        compressed_image = compressor(single_image_tensor).squeeze(dim=0)
                        output_image = compressed_image.tolist()
                        singleImageData = {'image' : output_image}
                        json.dump(singleImageData, newOutputFile)
                
                # set train / val / test (80 - 10 - 10)
                partition = None
                if counter % 10 == 1:
                    partition = "validation"
                elif counter % 10 == 2:
                    partition = "test"
                else:
                    partition = "train"

                # Add an entry to the dataLoaderData file
                singleImageJSONData = {'label' : shape,
                                    'filename' : singleImagePath,
                                    'partition' : partition} 
                dataLoaderData.append(singleImageJSONData)

                # Increment counter to make unique file names
                counter+=1

                # impliments truncation
                if trunc_len != None and trunc_len < counter:
                    break 
    
    # Write dataLoaderData to dataLoaderFile
    with open(f"{json_output_name}.json", 'w') as dataLoaderFile:
        json.dump(dataLoaderData, dataLoaderFile)

def convertStrokeVectorToBitmap(strokes, size=256):
    """
    Given a nested array of strokes, return a black and white bitmap of size x size
    """
    # Create a blank image with a white background
    image_size = (size, size)
    image = Image.new("L", image_size, 0)

    # Get a drawing context
    draw = ImageDraw.Draw(image)

    # Function to draw the strokes on the image
    def draw_stroke(draw, x_coords, y_coords):
        points = list(zip(x_coords, y_coords))
        draw.line(points, fill=255, width=2)

    # Process each stroke
    for x_coords, y_coords in strokes:
        draw_stroke(draw, x_coords, y_coords)

    # Convert the PIL image to a PyTorch tensor
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32)

    # Normalize the tensor to have 1s where the line is drawn and 0s elsewhere
    return ((image_tensor > 0).float()).tolist()

if __name__ == "__main__":
    list_of_files = list_files_in_folder("data")
    read_ndjson_file(list_of_files, "medium_test_data", "medium_test_data_json", trunc_len =10000, three_d = True, compress = True, pad_len = 8)
    
# tests both funcitons in dataFormatter 
# reads in smaple data, creates images, stores them in files, looks at the stored files, and displays the images 
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from dataFormatter import *
from dataManager import * 
import os

from utility_methods import list_files_in_folder

def plot_nparray(image, label, output_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray', aspect = "equal")  # Use a grayscale color map
    plt.axis("on")
    plt.colorbar()
    plt.text(5, 5, label, color='white', fontsize=12, backgroundcolor='black')
    plt.savefig(f"{output_path}")
    plt.close()

def test_dataFormatter():

    # read in the small file 
    read_ndjson_file(["SMALLfull-simplified-triangle.ndjson", "Smallfull-simplified-circle.ndjson"], "parsedData", "small_json_output", compress = True) 

    images_to_plot = [] # (image, label)

    # open the generated json file: 
    with open("small_json_output.json", 'r') as file:
        # Load the data from the file
        data = json.load(file)

        for obj in data: 
            label = obj['label']
            image_path = obj['filename']
            image_dict = None

            with open(image_path, 'r') as img_file:
                image_dict = json.load(img_file)

            image = image_dict['image']
            images_to_plot.append((np.array(image), label))

    # no longer have file open! 

    os.makedirs("testImages", exist_ok=True)
    counter = 0
    for image, label in images_to_plot:
        plot_nparray(image, label, f"testImages/dataFormatter{counter}")
        counter += 1

def test_dataFormatter_3d():
    # read in small file with 3d functionality 
    read_ndjson_file(["SMALLfull-simplified-triangle.ndjson", "Smallfull-simplified-circle.ndjson"], "parsedData", "small_json_output", three_d = True, compress = True)

    images_to_plot = []

    with open('small_json_output.json', 'r') as file:
        data = json.load(file)

        for obj in data:
            label = obj['label']
            image_path = obj['filename']
            image_dict = None

            with open(image_path, 'r') as img_file:
                image_dict = json.load(img_file)
            
            images = image_dict['image']
            images_to_plot.append((images, label)) # each element in here is a list of images 


            os.makedirs("testImages", exist_ok=True)
            macro_counter = 0
            for images, label in images_to_plot:
                image_counter = 0
                for image in images:
                    plot_nparray(np.array(image), label, f"testImages/df_imgno{macro_counter}_img{image_counter}")
                    image_counter += 1
                macro_counter += 1

def test_dataManager():
    # read raw data
    read_ndjson_file(["SMALLfull-simplified-triangle.ndjson", "Smallfull-simplified-circle.ndjson"], "parsedData", "small_json_output")
    # get pack to json file we created 
    path_to_json = 'jsonData/circle.data.json'
    data_directory = 'parsedData'


    train_partition = DataPartition(path_to_json, data_directory, 'train')
    test_partition = DataPartition(path_to_json, data_directory, 'train') # this would be test in real enviornment 

    manager = DataManager(train_partition, train_partition, train_partition) # default arguments for rest should be fine 

    count = 0
    batch_num = 0
    response_list = train_partition.possible_responses()

    for batch in manager.test():
        # batch size should be 4 here, unless otherwise specified (with some remainder at the end)
        features, responses = manager.features_and_response(batch)
        print(f"this is the feature shape: {features.shape}")
        print(f"this is the responses shape: {responses.shape}")

        for i, feature in enumerate(features):
            np_feature = feature.numpy()
            int_label = responses[i]
            plot_nparray(np_feature, response_list[int_label], f"testImages/dataManager_{batch_num}_{count}")
            count += 1
        
        batch_num += 1

def test_data_manager_3d():
    #list_of_files = list_files_in_folder("data")
    #read_ndjson_file(list_of_files, "small_test_data", "small_test_data_json", 20, three_d = True)

    path_to_json = "small_test_data_json.json"
    data_directory = "NA"

    train_partition = DataPartition(path_to_json, data_directory, 'train')
    val_partition = DataPartition(path_to_json, data_directory, 'validation')
    test_partition = DataPartition(path_to_json, data_directory, 'test')

    manager = DataManager(train_partition, val_partition, test_partition)

    print('train batch size:')
    for batch in manager.train(4):
        features, responses = manager.features_and_response(batch)
        print(features.shape)
        print(responses.shape)
        break

    print('validation batch size')
    for batch in manager.validation(4):
        features, responses = manager.features_and_response(batch)
        print(features.shape)
        print(responses.shape)
        break
    
    print('test batch size')
    for batch in manager.test(4):
        features, responses = manager.features_and_response(batch)
        print(features.shape)
        print(responses.shape)
        break


if __name__ == "__main__":
    test_dataFormatter_3d()












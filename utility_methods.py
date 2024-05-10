import matplotlib.pyplot as plt
from tqdm import tqdm 
import json
import os


# Sample data
data = []

def list_files_in_folder(folder_path):
    file_paths = []
    
    # Traverse through all the files and folders in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the full path of each file
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    
    return file_paths

def visualizeStrokeCount(listOfPathToRawData, trunc_len = None):
    """
    Given a path to JSON formatted training images in stroke form and a folder to output to,
    converts each image to a bitmap in JSON format and places it into its own file in 
    output folder. Also creates a dataLoader data.json file in Mark's style.
    """
    # Counter for output file naming
    counter = 0


    # Open raw data file
    for path in listOfPathToRawData:

        with open(path, 'r') as rawDataFile:
            for line in tqdm(rawDataFile, desc = f"Processing: {path}"):
                # Parse the JSON data from each line. Each line is an image
                record = json.loads(line)
                
                # Calculate the number of strokes
                data.append(len(record.get('drawing')))

                # Increment counter to make unique file names
                counter+=1

                # impliments truncation
                if trunc_len != None and trunc_len < counter:
                    break 


    # Plotting the histogram
    plt.hist(data, bins=10, color='blue', alpha=0.7, range=[0,20])

    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Example usage
    folder_path = '/Users/jacobcohen/cs381/Deep_Final_Project/data'  # Specify your folder path here
    files = list_files_in_folder(folder_path)
    visualizeStrokeCount(files)
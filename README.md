# Deep_Final_Project
Lola, Thomas, and Jacob

## File description
- ``dataFormatter.py``: Formats raw data from Google into json files that ``dataManager.py`` expects.
- ``dataManager.py``: Manages data for training, testing, and validating the model. Large image files stored on disk instead of RAM when required.
- ``frankenstein.py``: Trains Frankenstein (and the cnn only model with minor modifications).
- ``test_dataFormatter.py``: Suite of test functions for ``dataFormatter.py`` and ``dataManager.py`` that runs on smaller input.
- ``utility_methods.py``: Gets all the file names in a directory and visualizes the number of strokes in a dataset.
- ``button.py``: Button object for front-end.
- ``paintCanvas.py``: Canvas for users to draw on.
- ``medium_model_lstm``: The trained LSTM-based model.
- ``medium_model_cnn``: The trained cnn only based model.

## Important downloads from Drive BEFORE running files below
- Unzip ``medium_test_data.zip`` and ``data.zip`` into the main project folder
- https://drive.google.com/drive/folders/1MoVnZ6-MgbVXc3y-eWM2k0H8S0DgGPI7?usp=sharing
  
## Instructions to run ``paintCanvas.py``
- ``python paintCanvas.py``
- Should open up a white canvas with buttons at the bottom.
- Sketches capped at 8 strokes and the sketch will reset if you exceed this.
- Might be buggy, seemed to always crash on Thomas' computer :(.

## Instructions to run ``frankenstein.py``
- ``python frankenstein.py``
- Takes forever not recommended. We provide the trained models in ``medium_model_lstm`` and ``medium_model_cnn``.

## Instructions to run ``test_dataFormatter.py``
- ``python test_dataFormatter.py``
- Outputs stroke images into ``testImages`` from a small test json input files ``SMALLfull-simplified-circle.ndjson`` and ``SMALLfull-simplified-triangle.ndjson``
- Variety of additional test methods for the dataFormatter and dataManager for 2d and 3d images

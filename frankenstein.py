import torch
import torch.optim as optim
import torch.nn as nn
from dataManager import retrieve_image, nlog_softmax_loss, minibatch_training
from dataManager import *

class UnsqueezeDimension(nn.Module):
    def __init__(self, dimension = None):
        super().__init__()
        self.dimension = dimension
    
    def forward(self, x):
        return x.unsqueeze(dim = self.dimension)

class SqueezeDimension(nn.Module):
    def __init__(self, dimension=None):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return x.squeeze(dim=self.dimension)


class LSTM_ORDER(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        reordered = x.permute(0, 2, 1, 3, 4)
        return reordered.reshape(reordered.shape[0], reordered.shape[1], -1)

class LSTM_OUTPUT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output, _ = x # outputs hidden states as well, don't need those
        return output

class Flatten(nn.Module):
    """
    Flattens a tensor into a matrix. The first dimension of the input
    tensor and the output tensor should agree.
    
    For instance, a 3x4x5x2 tensor would be flattened into a 3x40 matrix.
    
    See the unit tests for example input and output.
    
    """   
    def __init__(self): 
        super().__init__()
    
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

def build_frankenstein(output_classes, dense_hidden_size = 128, LSTM_hidden_size = 256, channels_list = [16, 8, 4]): 
    model = nn.Sequential()
    model.add_module('unsqueeze1', UnsqueezeDimension(1))
    model.add_module('conv1', nn.Conv3d(in_channels = 1, out_channels = channels_list[0], kernel_size = (3,3,3), stride = 1, padding = 1))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))) # (BATCH, 8, 10, 32, 32) (with trucation, otherwise 128)
    model.add_module('conv2', nn.Conv3d(in_channels = channels_list[0], out_channels = channels_list[1], kernel_size = (3,3,3), stride = 1, padding = 1)) 
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))) # (BATCH, 4, 10, 16, 16) 
    model.add_module('conv3', nn.Conv3d(in_channels = channels_list[1], out_channels = channels_list[2], kernel_size = (3,3,3), stride = 1, padding = 1))
    model.add_module('relu3', nn.ReLU())
    model.add_module('pool3', nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))) # (BATCH, channels_list[2], 8 (truncation), 8, 8) - last 8x8 is image size at this point 
    model.add_module('re-order', LSTM_ORDER()) # (BATCH, 8 (truncation), channels_list[2] * 8 * 8)
    model.add_module('LSTM', nn.LSTM(input_size = channels_list[2] * 8 *8, hidden_size=LSTM_hidden_size, num_layers = 3, batch_first=True, dropout=0.0)) # input: (BATCH, L (length), input_size) --> inpust size = channels_list[2], 8, 8 
    model.add_module("LSTM output", LSTM_OUTPUT())   
    model.add_module('flatten2', Flatten()) # BATCH (omitted) * L (8) * channels_list[2] * 8 * 8
    model.add_module('linear1', nn.Linear(in_features = (8 * LSTM_hidden_size), out_features = dense_hidden_size)) # 8 is # frames, 64 is output image,  # might be error here - NEED BATCH SIZE? was 10 x 1024
    model.add_module('relu4', nn.ReLU()) 
    model.add_module('linear2', nn.Linear(in_features = dense_hidden_size, out_features = output_classes))
    return model

def build_cnn_only(output_classes, dense_hidden_size = 128, channels_list = [16, 8, 4]): 
    model = nn.Sequential()
    model.add_module('unsqueeze1', UnsqueezeDimension(1))
    model.add_module('conv1', nn.Conv3d(in_channels = 1, out_channels = channels_list[0], kernel_size = (3,3,3), stride = 1, padding = 1))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))) # (BATCH, 8, 10, 128, 128) or 32
    model.add_module('conv2', nn.Conv3d(in_channels = channels_list[0], out_channels = channels_list[1], kernel_size = (3,3,3), stride = 1, padding = 1)) 
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))) # (BATCH, 4, 10, 64, 64) or 16
    model.add_module('conv3', nn.Conv3d(in_channels = channels_list[1], out_channels = channels_list[2], kernel_size = (3,3,3), stride = 1, padding = 1))
    model.add_module('relu3', nn.ReLU())
    model.add_module('pool3', nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))) # (BATCH, 1, 10, 32, 32) or 8
    model.add_module('flatten2', Flatten())
    model.add_module('linear1', nn.Linear(in_features = 8 * 64 * channels_list[2], out_features = dense_hidden_size)) # 8 is # frames, 64 is output image (8x8), channels_list[2] is num channels coming in
    model.add_module('relu4', nn.ReLU()) 
    model.add_module('linear2', nn.Linear(in_features = dense_hidden_size, out_features = output_classes))
    return model

class Classifier:
    """
    Allows the trained MODEL to be saved to disk and loaded back in.

    You can call a Classifier instance as a function on an image filename
    to obtain a probability distribution over whether it is a zebra.
    
    """
    
    def __init__(self, net, catagories, 
                 dense_hidden_size, LSTM_hidden_size, channels_list):
        # the net loaded in needs to be the same
        self.net = net
        self.catagories = catagories
        self.dense_hidden_size = dense_hidden_size
        self.channels_list = channels_list
        self.lstm_hidden_size = LSTM_hidden_size

 
    def __call__(self, img_filename):
        self.net.eval()
        image = None
        if isinstance(img_filename, str):
            image = retrieve_image(img_filename)
        else: # is a tensor already
            image = img_filename
        inputs = image.float().unsqueeze(dim=0)
        outputs = torch.softmax(self.net(inputs), dim=1)
        result = dict()
        for i, category in enumerate(self.catagories): # maybe error here?
            result[category] = outputs[0][i].item()
        return result

    def save(self, filename):
        torch.save(self.net.state_dict(), f"{filename}")
            
    def load(self, config_file):
        return self.net.load_state_dict(torch.load(config_file))

def run(data_config, n_epochs, channels_list, dense_hidden_size, LSTM_hidden_size):    
    """
    Runs a training regime for a CNN.

    data_config is the json file with the data headers
    
    """
    train_set = DataPartition(data_config, 'train') # data config is a json file
    val_set = DataPartition(data_config, 'validation')
    test_set = DataPartition(data_config, 'test')

    manager = DataManager(train_set, val_set, test_set)
    loss = nlog_softmax_loss 
    learning_rate = .001
    #image_width = 64

    # replace with neural net of your choice
    net = build_cnn_only(output_classes = len(val_set.possible_responses()), dense_hidden_size = dense_hidden_size, channels_list = channels_list) # pass through LSTM_hidden_size if LSTM
    #net = build_frankenstein(output_classes = len(val_set.possible_responses()), dense_hidden_size = dense_hidden_size, channels_list = channels_list, LSTM_hidden_size=LSTM_hidden_size)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  
    best_net, monitor = minibatch_training(net, manager, 
                                           batch_size=32, n_epochs=n_epochs, 
                                           optimizer=optimizer, loss=loss)
    classifier = Classifier(best_net, val_set.possible_responses(), 
                            dense_hidden_size, channels_list, None)
    return classifier, monitor


if __name__ == "__main__":
    # runs the model 
    classifier, monitor = run("medium_test_data_json.json", 40, channels_list = [32, 16, 8], dense_hidden_size = 256, LSTM_hidden_size=256)
    classifier.save("medium_model_cnn")
    

    # test_set = DataPartition("small_test_data_json.json", 'test')
    # val_set = DataPartition("small_test_data_json.json", 'validation')
    # train_set = DataPartition("small_test_data_json.json", 'train')
    # manager = DataManager(train_set, val_set, test_set)

    # net = build_cnn_only(output_classes = len(train_set.possible_responses()), dense_hidden_size = 256, channels_list = [32, 16, 8])
    # new_classifier = Classifier(net, test_set.possible_responses(), None, None, None)
    # new_classifier.load("medium_model_lstm")


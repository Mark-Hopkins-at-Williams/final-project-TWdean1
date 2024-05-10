# data management stuff. yay 
import torch
from torch.nn import Parameter
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from torchvision import io, transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from copy import deepcopy
from tqdm import tqdm
import time
import sys
import json
import os

# TODO: data_dir not used -- go through and eliminate

__all__ = ["DataPartition", "DataManager"]

def retrieve_image(filepath):
    """
    returns a torch tensor that is bitmaped version of image
    """

    image = None
    with open(filepath, 'r') as f:
        image = json.load(f)
    
    image = image['image']
    return torch.tensor(image)
    
class DataPartition(Dataset):

    def __init__(self, json_file, partition): #resize_width=128
        """
        Creates a DataPartition from a JSON configuration file.
        
        - json_file is the filename of the JSON config file. NOTE: MUST HAVE 1 JSON FILE (need to aggregate)
        - data_dir is the directory where the images are stored.
        - partition is a string indicating which partition this object
                    represents, i.e. "train" or "test"
        - resize_width indicates the dimensions to which each image will
                    be resized (the images are resized to a square WxW
                    image)
        
        """  
        # get json lables    
        with open(json_file, 'r') as f:
            self.lables = json.load(f)
        
        self.lables = [p for p in self.lables if p['partition'] == partition]

    def __len__(self):
        """
        Returns the number of data (datums) in this DataPartition.
        
        """
        return len(self.lables)

    def __getitem__(self, i):
        """
        Converts the ith datum into the following dictionary and then
        returns it:
            
            {'image': retrieve_image(img_filename), 
             'response': datum['label'], 
             'filename': datum['filename'] }
        
        """  
        datum = self.lables[i]
        return {"image": retrieve_image(f"{datum['filename']}"), # NOTE: This is now 3d# "{self.data_dir}/ (removed from path) #self.resize_width (removed)
                "response": datum["label"],
                "filename": datum["filename"]}
   
    def possible_responses(self):
        """
        Returns an alphabetically sorted list of response values (labels)
        found in this DataPartition -- all the distinct values associated with 
        the key "response" in any datum.
        
        """
        uniques = {datem["label"] for datem in self.lables}
        uniques = list(uniques)
        uniques.sort()
        return uniques

class DataManager:
    
    def __init__(self, train_partition, validation_partition, test_partition, 
                 features_key = 'image', response_key='response'):
        """
        Creates a DataManager from a JSON configuration file. The job
        of a DataManager is to manage the data needed to train and
        evaluate a neural network.
        
        - train_partition is the DataPartition for the training data.
        - test_partition is the DataPartition for the test data.
        - features_key is the key associated with the features in each
          datum of the data partitions, i.e. train_partition[i][features_key]
          should be the ith feature tensor in the training partition.
        - response_key is the key associated with the response in each
          datum of the data partitions, i.e. train_partition[i][response_key]
          should be the ith response tensor in the training partition.
        
        """
        self.train_set = train_partition
        self.validation_set = validation_partition
        self.test_set = test_partition
        self.features_key = features_key
        self.response_key = response_key
        
        # create integers for possible responses 
        self.response_dict = dict()
        for i, response in enumerate(self.train_set.possible_responses()):
            self.response_dict[response] = i
        
    def train(self, batch_size):
        """
        Returns a torch.DataLoader for the training examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
        
        - batch_size is the number of desired training examples per batch
        
        """
        return DataLoader(self.train_set, batch_size=batch_size,
                        sampler=RandomSampler(self.train_set))
    
    def validation(self, batch_size):
        """
        Returns a torch.DataLoader for the validation examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
        
        - batch_size is the number of desired training examples per batch

        """
        return DataLoader(self.validation_set, batch_size = batch_size, 
                            sampler = SequentialSampler(self.validation_set))

    def test(self, batch_size): # legacy of marks --> any reason for this?
        """
        Returns a torch.DataLoader for the test examples. The returned
        DataLoader can be used as follows:
            
            for batch in data_loader:
                # do something with the batch
                
        """
        return DataLoader(self.test_set, batch_size=batch_size, # possible bug
                          sampler=SequentialSampler(self.test_set))

    
    def features_and_response(self, batch):
        """
        Converts a batch obtained from either the train or test DataLoader
        into a feature tensor and a response tensor.
        
        The feature tensor returned is just batch[self.features_key].
        
        To build the response tensor, one starts with batch[self.response_key],
        where each element is a "response value". Each of these response
        values is then mapped to the index of that response in the sorted set of
        all possible response values. The resulting tensor should be
        a LongTensor.

        The return value of this function is:
            feature_tensor, response_tensor
        
        See the unit tests in test.py for example usages.
        
        """
        feature_tensor = batch[self.features_key]
        response_tensor = torch.tensor([self.response_dict[response] for response in batch[self.response_key]]).long()
        return feature_tensor, response_tensor


    def evaluate(self, classifier, partition, batch_size):
        """
        Given a classifier that maps a feature tensor to a response
        tensor, this evaluates the classifier on the specified data
        partition ("train" or "test") by computing the percentage of
        correct responses.
        
        See the unit test ```test_evaluate``` in test.py for expected usage.
        
        """
        
        if partition == "train" :
            data_loader = self.train(batch_size) 
        elif partition == "validation":
            data_loader = self.validation(batch_size)
        else:
            data_loader = self.test(batch_size)
        
        num_correct = 0
        total = 0
        for batch in data_loader: 
            feature_tensor, response_tensor = self.features_and_response(batch)
            predicted = classifier(feature_tensor)
            # want index of max element for each thing in predicted 
            refined_predicted = torch.argmax(predicted, dim=1) 
            num_correct += (refined_predicted == response_tensor).sum().item()
            total += response_tensor.numel()
        
        return num_correct / total


class TrainingMonitor:
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def start(self, n_batches):
        print("=== STARTING TRAINING ===")
        self.training_start_time = time.time()
        self.n_batches = n_batches
        self.print_every = n_batches // 10   
        self.train_profile = []

    def start_epoch(self, epoch):
        sys.stdout.write("Epoch {}".format(epoch))
        self.running_loss = 0.0
        self.start_time = time.time()
        self.total_train_loss = 0

    def report_batch_loss(self, epoch, batch, loss):
        self.total_train_loss += loss
        if self.verbose:
            self.running_loss += loss
            if (batch + 1) % (self.print_every + 1) == 0:               
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (batch+1) / self.n_batches), 
                        self.running_loss / self.print_every, time.time() - self.start_time))
                self.running_loss = 0.0
                self.start_time = time.time()                    
            
    def report_accuracies(self, epoch, train_acc, dev_acc):
        self.train_profile.append((train_acc, dev_acc))
        epoch_time = time.time() - self.start_time
        if train_acc is not None:
            print("Train accuracy = {:.2f}".format(train_acc))
        if dev_acc is not None:
            print("[{:.2f}sec] validation loss = {:.2f}; validation accuracy = {:.2f} ".format(
                    epoch_time, self.total_train_loss, dev_acc))

    def training_profile_graph(self):
        return [[x[0] for x in self.train_profile], 
                [x[1] for x in self.train_profile]]

    def stop(self):
        print("Training finished, took {:.2f}s".format(
                time.time() - self.training_start_time))

    @staticmethod
    def plot_average_and_max(monitors, description=""):
        valuelists = [monitor.training_profile_graph()[1] for monitor in monitors]
        values = [sum([valuelist[i] for valuelist in valuelists])/len(valuelists) for 
                  i in range(len(valuelists[0]))]
        overall = list(zip(range(len(values)), values))
        x = [el[0] for el in overall]
        y = [el[1] for el in overall]
        plt.plot(x,y, label='average' + description)
        values = [max([valuelist[i] for valuelist in valuelists]) for 
                  i in range(len(valuelists[0]))]
        overall = list(zip(range(len(values)), values))
        x = [el[0] for el in overall]
        y = [el[1] for el in overall]
        plt.plot(x,y, label='max' + description)
        plt.xlabel('epoch')
        plt.ylabel('test accuracy')
        plt.legend()

    @staticmethod
    def plot_average(monitors, description=""):
        valuelists = [monitor.training_profile_graph()[1] for monitor in monitors]
        values = [sum([valuelist[i] for valuelist in valuelists])/len(valuelists) for 
                  i in range(len(valuelists[0]))]
        overall = list(zip(range(len(values)), values))
        x = [el[0] for el in overall]
        y = [el[1] for el in overall]
        plt.plot(x,y, label='average' + description)
        plt.xlabel('epoch')
        plt.ylabel('test accuracy')
        plt.legend()



def nlog_softmax_loss(X, y):
    """
    A loss function based on softmax, described in colonels2.ipynb. 
    X is the (batch) output of the neural network, while y is a response 
    vector.
    
    See the unit tests in test.py for expected functionality.
    
    """    
    smax = torch.softmax(X, dim=1)
    correct_probs = torch.gather(smax, 1, y.unsqueeze(1))
    nlog_probs = -torch.log(correct_probs)
    return torch.mean(nlog_probs) 


def minibatch_training(net, manager, batch_size, 
                       n_epochs, optimizer, loss):
    """
    Trains a neural network using the training partition of the 
    provided DataManager.
    
    Arguments
    - net: the Module you want to train.
    - manager: the DataManager
    - batch_size: the desired size of each minibatch
    - n_epochs: the desired number of epochs for minibatch training
    - optimizer: the desired optimization algorithm to use (should be an 
                 instance of torch.optim.Optimizer)
    - loss: the loss function to optimize
    
    """
    monitor = TrainingMonitor()
    train_loader = manager.train(batch_size)
    best_accuracy = float('-inf')
    best_net = None
    monitor.start(len(train_loader))
    for epoch in range(n_epochs):
        monitor.start_epoch(epoch)
        net.train() # puts the module in "training mode", e.g. ensures
                    # requires_grad is on for the parameters
        for i, data in tqdm(enumerate(train_loader, 0)):
            features, response = manager.features_and_response(data)
            optimizer.zero_grad()
            output = net(features)
            batch_loss = loss(output, response)
            batch_loss.backward()
            optimizer.step()
            monitor.report_batch_loss(epoch, i, batch_loss.data.item())            
        net.eval() # puts the module in "evaluation mode", e.g. ensures
                   # requires_grad is off for the parameters
        dev_accuracy = manager.evaluate(net, "validation", batch_size) 
        monitor.report_accuracies(epoch, None, dev_accuracy)
        if dev_accuracy >= best_accuracy:
            best_net = deepcopy(net)     
            best_accuracy = dev_accuracy
    monitor.stop()
    return best_net, monitor


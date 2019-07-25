import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



#pragma: coderesponse template name="cnn"
class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        ch1 = 64
        ch2 = ch1 * 2
        ch3 = ch2 * 2
        i = 3
        j = int(((int(((int(((42 - i*3 + 1*3)-1)/2) - i*3 + 1*3)-1)/2) - i*3 + 1*3)-1)/2)

        k = int(((int(((int(((28 - i*3 + 1*3)-1)/2) - i*3 + 1*3)-1)/2) - i*3 + 1*3)-1)/2)
        
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(1, ch1, (i, i))
        self.conv2 = nn.Conv2d(ch1, ch1, (i, i))
        self.conv3 = nn.Conv2d(ch1, ch1, (i, i))
        self.conv4 = nn.Conv2d(ch1, ch2, (i, i))
        self.conv5 = nn.Conv2d(ch2, ch2, (i, i))
        self.conv6 = nn.Conv2d(ch2, ch2, (i, i))
        self.conv7 = nn.Conv2d(ch2, ch2*2, (i, i))
        self.conv8 = nn.Conv2d(ch2, ch2*2, (i, i))
        self.conv9 = nn.Conv2d(ch2, ch2*2, (i, i))        
        
        self.dropout = nn.Dropout(0.5)
        self.fc11 = nn.Linear(j*k*ch3, j*k*ch3)
        self.fc12 = nn.Linear(j*k*ch3, j*k*ch3)
        self.fc21 = nn.Linear(j*k*ch3, j*k*ch3)
        self.fc22 = nn.Linear(j*k*ch3, j*k*ch3)
        self.final1 = nn.Linear(j*k*ch3, 10)
        self.final2 = nn.Linear(j*k*ch3, 10)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        #input image 42x28
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpool(x)               
        #38x24x32
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.maxpool(x)      
        #18x11x32

        #x = self.dropout(x)
        #6x3x64
        x = self.flatten(x)
        
        x1 = self.fc11(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc12(x)
        x1 = self.relu(x1)
        x1 = self.final1(x1)
        
        x2 = self.fc21(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.fc22(x2)
        x2 = self.relu(x2)
        x2 = self.final2(x)
        
        out_first_digit = x1
        out_second_digit = x2
        

        return out_first_digit, out_second_digit
#pragma: coderesponse end

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()

import csv

from Partial_observation_model_functions import *
from torch.utils.data import DataLoader
import argparse


number_samples = [5000]
learning_rate = 0.5
rate = '5'
train_size = [10, 50, 100, 300, 500]
validation_size = 200
test_size = [1000]

num_epochs = 30


batch_size = 500
time_step = [1, 5]

# main_20(number_samples, learning_rate, rate, train_ratio, validation_ratio,
#         test_ratio, num_epochs, batch_size, time_step)


main_kro(number_samples, learning_rate, rate, train_size, validation_size,
         test_size, num_epochs, batch_size, time_step)




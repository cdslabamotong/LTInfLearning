from full_observation.Full_observation_functions import *
import numpy as np
import math
import torch.nn as nn
from torch.utils.data import Dataset
import time
import networkx as nx
import torch
import torch.nn.functional
import csv
from torch.utils.data import DataLoader


def add_attr_on_graph(graph, number_cascades, decimal_number):

    for i in graph.nodes:
        node_attr = {'status': 0}
        for j in range(1, number_cascades + 1):
            node_attr[f'threshold_{j}'] = Decimal(random.uniform(0, 1)).quantize(Decimal('.001'), rounding=ROUND_UP)
        graph.nodes[i]['attr'] = node_attr
    for edge in graph.edges:
        edge_attr = {}
        for k in range(1, number_cascades + 1):
            degree_in = graph.degree[edge[1]]
            edge_attr[f'weight_{k}'] = Decimal(1 / (degree_in + k)).quantize(Decimal('0.001'), rounding=ROUND_UP)
        graph.edges[edge[0], edge[1]]['attr'] = edge_attr
    return graph


def load_kro_graph(num_cascades, decimal_number):

    file_path = '/home/cds/Documents/project_influence_function_learning/data/Kro_data/kro_diffusionModel.txt'
    n_nodes = 1024
    file = open(file_path, 'r')
    lines = file.readlines()
    edge_list = []
    for line in lines:
        li = line.split(' ')
        node_u = int(li[0])
        node_v = int(li[1])
        if node_u < n_nodes+1 and node_v < n_nodes+1:
            edge = (node_u, node_v)
            edge_list.append(edge)
    nodes_list = []
    for node in range(n_nodes):
        nodes_list.append(node)
    g = nx.DiGraph()
    g.add_nodes_from(nodes_list)
    g.add_edges_from(edge_list)
    g_init = add_attr_on_graph(g, num_cascades, decimal_number)
    nodes_attr = nx.get_node_attributes(g_init, 'attr')
    threshold_list = []
    for i in range(n_nodes):
        for j in range(num_cascades):
            threshold = float(nodes_attr[i][f'threshold_{j+1}'])
            threshold_list.append(threshold)
    threshold_array = np.array(threshold_list)

    return g_init, threshold_array


def load_er_graph(num_cascades, decimal_number):

    g = nx.erdos_renyi_graph(n=3000, p=0.5, directed=True)
    n_nodes = g.number_of_nodes()
    g_init = add_attr_on_graph(g, num_cascades, decimal_number)
    nodes_attr = nx.get_node_attributes(g_init, 'attr')
    threshold_list = []
    for i in range(n_nodes):
        for j in range(num_cascades):
            threshold = float(nodes_attr[i][f'threshold_{j+1}'])
            threshold_list.append(threshold)
    threshold_array = np.array(threshold_list)

    return g_init, threshold_array


def load_20_nodes_graph(num_cascades, decimal_number):
    n_nodes = 20
    nodes_list = []
    for node in range(n_nodes):
        nodes_list.append(node)
    edge_list = [(1, 4), (2, 4), (3, 2), (3, 4), (0, 3), (0, 4), (0, 2), (1, 2), (1, 7), (1, 9), (0, 8), (9, 5),
                 (9, 3), (7, 3), (7, 9), (6, 8), (6, 5), (5, 0), (5, 1), (11, 13), (9, 13), (11, 14), (10, 6),
                 (9, 12), (8, 12), (0, 10), (1, 11), (6, 11), (7, 10), (13, 14), (12, 7), (14, 17), (16, 13), (18, 2),
                 (19, 0), (18, 15), (12, 19), (13, 18), (16, 7), (18, 10), (15, 5), (17, 5), (19, 6), (17, 8)]

    g = nx.DiGraph()
    g.add_nodes_from(nodes_list)
    g.add_edges_from(edge_list)
    g_init = add_attr_on_graph(g, num_cascades, decimal_number)
    nodes_attr = nx.get_node_attributes(g_init, 'attr')
    threshold_list = []
    for i in range(n_nodes):
        for j in range(num_cascades):
            threshold = float(nodes_attr[i][f'threshold_{j+1}'])
            threshold_list.append(threshold)
    threshold_array = np.array(threshold_list)
    # print(threshold_array)
    return g_init, threshold_array


# def data_preparation_for_partial_samples(step0, samples_set, num_samples, num_nodes, num_cascades, threshold_array):
#     output = []
#     for each_sample in samples_set:
#         output.append(each_sample[-1][-1])
#
#     x_array = np.array(step0)
#     y_array = np.array(output)
#
#     x_array = np.reshape(x_array, (num_samples, num_nodes * num_cascades, 1))
#     threshold_array = np.reshape(threshold_array, (num_nodes * num_cascades, 1))
#     threshold_array = threshold_array * (-1)
#     x_array_final = np.zeros((num_samples, num_nodes * num_cascades, 2))
#     for i in range(num_samples):
#         x_array_final[i] = np.hstack((x_array[i], threshold_array))
#
#     x_array_final = np.reshape(x_array_final, (num_samples, 2 * num_nodes * num_cascades))
#     # print(x_array_final)
#     # print(x_array_final.shape)
#
#     y_array = np.reshape(y_array, (num_samples, num_nodes*num_cascades))
#
#     return x_array_final, y_array


def data_preparation_for_partial_samples(step0, samples_set, num_samples, num_nodes, num_cascades):
    output = []
    for each_sample in samples_set:
        output.append(each_sample[-1][-1])

    x_array = np.array(step0)
    y_array = np.array(output)

    x_array = np.reshape(x_array, (num_samples, num_nodes * num_cascades))
    y_array = np.reshape(y_array, (num_samples, num_nodes * num_cascades))

    return x_array, y_array


def generate_partial_sample_set(graph, num_samples, num_nodes, num_cascades, threshold_array):
    print('Begin Generating Samples')
    sample_set = []
    count = 0
    step_0 = []
    for sample in range(num_samples):
        count += 1

        graph_step_0, initial_binary_mode = graph_at_step_0(graph, num_cascades)
        one_sample, t_step = diffusion_process_one_sample(graph_step_0, num_cascades, num_nodes)

        print(f'sample{sample}:')
        print(t_step)

        sample_set.append(one_sample)
        step_0.append(initial_binary_mode)

    x_samples, y_samples = data_preparation_for_partial_samples(step_0, sample_set, num_samples, num_nodes, num_cascades)

    print('End Generating Samples')

    return x_samples, y_samples


def for_torch_where_inplace(tensor1, fc):

    x, y = tensor1.shape
    tensor2 = torch.zeros(x, y)

    for x_idx in range(x):
        for y_idx in range(y):
            tensor2[x_idx][y_idx] = tensor1[x_idx][y_idx].item() + torch.transpose(fc, 0, 1)[y_idx+(y_idx + 1)][y_idx].item()
    tensor2 = tensor2.type(torch.FloatTensor)

    return tensor2


def matrix_in_comparison_part_3_cascade(num_cascades, num_nodes):

    y = num_nodes * num_cascades

    weights_1 = np.identity(y, dtype=int)
    weights_2 = np.zeros((y, num_nodes*(num_cascades-1)), dtype=int)
    weights_3 = np.identity(num_nodes*(num_cascades-1), dtype=int)
    weights_4 = np.zeros((num_nodes*(num_cascades-1), num_nodes), dtype=int)

    for i in range(num_nodes):
        weights_1[i*3+1][i*3] = -1

        weights_2[i*3][i*2] = 1
        weights_2[i*3+1][i*2] = 1
        weights_2[i*3+2][i*2+1] = 1

        weights_3[i*(num_cascades-1) + 1][i*(num_cascades-1)] = -1

        weights_4[2*i][i] = 1
        weights_4[2*i + 1][i] = 1

    # weights_1 = torch.from_numpy(weights_1).type(torch.FloatTensor)
    # weights_2 = torch.from_numpy(weights_2).type(torch.FloatTensor)
    # weights_3 = torch.from_numpy(weights_3).type(torch.FloatTensor)
    # weights_4 = torch.from_numpy(weights_4).type(torch.FloatTensor)

    return weights_1, weights_2, weights_3, weights_4


def matrix_one(value, num_cascades):
    matrix = np.ndarray((num_cascades, num_cascades), dtype=int)

    if value == 1:
        matrix = np.identity(num_cascades, dtype=int)
    if value == 0:
        matrix = np.zeros((num_cascades, num_cascades), dtype=int)

    return matrix


def sub_matrix(graph, num_cascades):
    a = nx.adjacency_matrix(graph).todense()
    x, y = a.shape
    final_list = []

    for i in range(x):

        matrix_x = matrix_one(a[i, 0], num_cascades)

        for j in range(1, y):
            matrix_one_rest = matrix_one(a[i, j], num_cascades)
            matrix_x = np.insert(matrix_x, [j*num_cascades], matrix_one_rest, axis=1)
        final_list.append(matrix_x)

    final_array = final_list[0]
    for ele_idx in range(1, x):
        final_array = np.insert(final_array, [ele_idx*num_cascades], final_list[ele_idx], axis=0)

    return final_array


# def generate_adjacency_matrix(graph, num_nodes, num_cascades):
#     n_cascade_array = sub_matrix(graph, num_cascades)
#     # print(n_cascade_array)
#     # print(n_cascade_array.shape)
#     minus_array = np.identity(num_nodes*num_cascades, dtype=int)
#     final_array = np.concatenate((n_cascade_array, minus_array), axis=1)
#     final_array = final_array.reshape((num_cascades*num_nodes*2, num_cascades*num_nodes))
#     # print(final_array)
#     # print(final_array.shape)
#
#     return final_array


def generate_adjacency_matrix(graph, num_nodes, num_cascades):
    n_cascade_array = sub_matrix(graph, num_cascades)
    # minus_array = np.identity(num_nodes*num_cascades, dtype=int)
    # final_array = np.concatenate((n_cascade_array, minus_array), axis=1)
    final_array = n_cascade_array.reshape((num_cascades*num_nodes, num_cascades*num_nodes))

    return final_array


def choose_experiment_data_set(num_samples, valid_size, train_size, test_size, x_samples, y_samples):

    # train_size = int(num_samples * train_ratio)
    # valid_size = int(num_samples * valid_ratio)
    # test_size = int(num_samples * test_ratio)

    x_sample = np.split(x_samples, [valid_size])
    x_valid = x_sample[0]
    x_sample_2 = x_sample[1]
    np.random.shuffle(x_sample_2)

    x_train_test = np.split(x_sample_2, [train_size])
    x_train = x_train_test[0]
    x_test = np.split(x_train_test[1], [test_size])[0]

    y_sample = np.split(y_samples, [valid_size])
    y_valid = y_sample[0]

    y_train_test = np.split(y_sample[1], [train_size])
    y_train = y_train_test[0]
    y_test = np.split(y_train_test[1], [test_size])[0]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def choose_random_samples(num_samples, ratio, x_sample, y_sample):
    number_rows = x_sample.shape[0]
    # print(number_rows)
    # print(num_samples*ratio)
    random_indices = np.random.choice(number_rows, size=int(num_samples*ratio), replace=True)
    x_random_rows = x_sample[random_indices, :]
    y_random_rows = y_sample[random_indices, :]

    return x_random_rows, y_random_rows


def max_function_matrix(num_nodes, num_cascades):
    a = torch.ones(num_nodes*num_cascades, num_nodes).type(torch.FloatTensor)

    for j in range(num_nodes):
        a[j*num_cascades][j] = 1.
        a[j*num_cascades + 1][j] = 1.
        a[j*num_cascades + 2][j] = 1.
    a.type(torch.FloatTensor)
    return a

# def training_loop(model, criterion, optimizer, train_loader, valid_loader, num_epoch, training_log_file):
#     log_file = open(training_log_file, 'a')
#     start_time = time.time()
#     n_total_step = len(train_loader)
#     n_total_valid_step = len(valid_loader)
#     for epoch in range(num_epoch):
#         print('epoch:', epoch)
#
#         # Set to training
#         for i, (input_, output_) in enumerate(train_loader):
#             # Forward pass and loss
#             batch_start_time = time.time()
#             # input_ = input_.cuda()
#             # output_ = output_.cuda()
#
#             # zero grad before new step
#             optimizer.zero_grad()
#
#             y_pred = model(input_)
#             train_loss = criterion(y_pred, output_)
#
#             # Backward pass and update
#             train_loss.backward()
#             optimizer.step()
#
#             if (i + 1) % 10 == 0:
#                 print(f'Epoch {epoch + 1}, Batch: {i + 1} Train Loss: {train_loss.item():.4f}')
#                 log_file.write(f'Epoch {epoch + 1}, Batch: {i + 1}Train Loss: {train_loss.item():.4f}'+'\n')
#             one_batch_time = time.time() - batch_start_time
#             print(f'one_batch_time: {one_batch_time}')
#             # log_file.write(f'one_batch_time: {one_batch_time}' + '\n')
#
#         for j, (input_, output_) in enumerate(valid_loader):
#             # Forward pass and loss
#
#             # input_ = input_.cuda()
#             # output_ = output_.cuda()
#
#             # zero grad before new step
#             optimizer.zero_grad()
#
#             y_pred = model(input_)
#             valid_loss = criterion(y_pred, output_)
#
#             print(f'Epoch: {epoch + 1}, Batch: {j + 1}, Train Loss: {valid_loss.item():.4f}')
#             log_file.write(f'Epoch: {epoch + 1}, Batch: {j + 1}, Train Loss: {valid_loss.item():.4f}' + '\n')
#             log_file.write('\n')
#     train_time = time.time() - start_time
#     print(f'total_train_time: {train_time}')
#     log_file.write(f'total_train_time: {train_time}')
#     log_file.close()
#     return model


# Define NewLinear
class MyLinearLayer(nn.Module):
    """ Custom Linear layer """
    def __init__(self, size_in, size_out, adjacency_matrix):

        super(MyLinearLayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out
        self.adj_matrix = adjacency_matrix
        # weights = torch.Tensor(self.size_in, self.size_out)

        self.weights = nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        nn.init.uniform_(self.weights, a=0., b=1.)

    def forward(self, x):

        out = torch.matmul(x, self.adj_matrix * self.weights)

        return out


# Define constrains on weight:
class UnitNormClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weights'):
            w = module.weights.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))


# define dataset
class PartialObservationData(Dataset):
    def __init__(self, sample_split_list, set_type):
        # data loading

        self.x_data = torch.from_numpy(sample_split_list[set_type]).type(torch.FloatTensor)
        self.y_data = torch.from_numpy(sample_split_list[set_type + 3]).type(torch.FloatTensor)
        self.data_size = int(self.x_data.shape[0])

    def __getitem__(self, index):
        # dataset[0]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.data_size


# Define Model
# class MultiCascades3(torch.nn.Module):
#     def __init__(self, decimal_number, size_in, size_out, adjacent_matrix, num_nodes, num_cascades):
#         super(MultiCascades3, self).__init__()
#
#         self.decimal_number = decimal_number
#         self.input_feature = size_in
#         self.output_feature = size_out
#         # self.m1 = m_1
#         # self.m2 = m_2
#         # self.m3 = m_3
#         # self.m4 = m_4
#         self.adj_max = adjacent_matrix
#         self.nodes = num_nodes
#         self.cascades = num_cascades
#
#         self.fc_1 = nn.Linear(self.input_feature, self.output_feature, bias=False)
#         self.fc_1.weight = nn.Parameter(torch.transpose(self.adj_max * (torch.rand(self.input_feature, self.output_feature)), 0, 1))
#         # print(self.fc_1.weight.shape)
#         # self.fc_2 = nn.Linear(self.output_feature, self.output_feature, bias=False)
#         # self.fc_3 = nn.Linear(self.output_feature, self.output_feature/3 * 2, bias=False)
#         # self.fc_4 = nn.Linear(self.output_feature/3 * 2, self.output_feature/3 * 2, bias=False)
#         # self.fc_4 = nn.Linear(self.output_feature/3 * 2, self.output_feature/3, bias=False)
#         #
#         # with torch.no_grad():
#         #     self.fc_2.weight = nn.Parameter(self.m1)
#         #     self.fc_3.weight = nn.Parameter(self.m2)
#         #     self.fc_4.weight = nn.Parameter(self.m3)
#         #     self.fc_5.weight = nn.Parameter(self.m4)
#
#     def forward(self, x):
#
#         data_size = x.shape[0]
#
#         # for i in range(20):
#
#         out_2 = self.fc_1(x)
#
#
#         # print(f'out_2:{out_2.shape}')
#         # print(out_2)
#         # return summation weights minus threshold
#         fc_6 = nn.Threshold(threshold=0, value=0, inplace=False)
#         out_3 = fc_6(out_2).type(torch.FloatTensor)
#
#
#         # print(f'out_3:{out_3.shape}')
#         # print(out_3)
#         # return summation weights or 0
#         inplace = for_torch_where_inplace(out_3, self.fc_1.weight)
#
#
#         # print(f'inplace:{inplace.shape}')
#         # print(inplace)
#         out_4 = torch.where((out_3 > 0), inplace, torch.zeros((data_size, self.output_feature)).type(torch.FloatTensor))
#
#
#         # print(f'out_4:{out_4.shape}')
#         # print(out_4)
#         out_5 = torch.reshape(out_4, [data_size, self.nodes, self.cascades])
#
#
#         # print(f'out_5:{out_5.shape}')
#         # print(out_5)
#
#         # max:
#         out_5_values, out_5_index = torch.max(out_5, dim=2)
#
#
#         # print(f'out_5_value:{out_5_values.shape}')
#         # print(out_5_values)
#         # recover:
#         out_5_values_1 = torch.repeat_interleave(out_5_values, self.cascades, dim=1)
#
#
#         # print(f'out_5_value_1:{out_5_values_1.shape}')
#         # print(out_5_values_1)
#         out_5_values_2 = torch.where((out_5_values_1 == 0.), -1., out_5_values_1.type(torch.DoubleTensor))
#
#
#         # print(f'out_5_value_2:{out_5_values_2.shape}')
#         # print(out_5_values_2)
#
#         out_6 = torch.where((out_4 == out_5_values_2), 1., 0.)
#         out_6.type(torch.FloatTensor)
#         out_6.requires_grad = True
#
#         print(out_6.grad_fn)
#         return out_6
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class MultiCascades3(torch.nn.Module):
    def __init__(self, decimal_number, size_in, size_out, adjacent_matrix, num_nodes, num_cascades, threshold, time_step):
        super(MultiCascades3, self).__init__()

        self.decimal_number = decimal_number
        self.input_feature = size_in
        self.output_feature = size_out
        self.adj_max = adjacent_matrix
        self.nodes = num_nodes
        self.cascades = num_cascades
        self.threshold = threshold
        self.timestep = time_step

        self.fc_1 = nn.Linear(self.input_feature, self.output_feature, bias=False)
        self.fc_2 = nn.PReLU()
        self.fc_3 = nn.Sigmoid()

    def one_step(self, input_, threshold_tensor):
        out_2 = self.fc_1(torch.matmul(input_, self.adj_max))
        out_3 = out_2 - threshold_tensor
        out_4 = self.fc_2(out_3)

        out_5 = self.fc_3(out_4)
        return out_5

    def forward(self, x):
        data_size = x.shape[0]

        threshold_tensor = self.threshold.repeat(data_size)
        threshold_tensor = threshold_tensor.reshape(data_size, self.input_feature)
        # print(threshold_tensor.shape)
        out_5 = torch.zeros(data_size, self.output_feature)
        for i in range(self.timestep):
            print(f'time_step:{i}')
            # print(f'x: {x}')
            out_5 = self.one_step(x, threshold_tensor)
            x = out_5
            # print(f'out_5: {out_5}')

        return out_5


def main_20(number_samples, learning_rate, rate, train_ratio, validation_ratio, test_ratio, num_epochs, batch_size, time_step):
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_printoptions(precision=4)

    # number_samples = [5000]
    # learning_rate = 0.01
    # rate = '01'

    for n_sample in number_samples:
        results_filepath = f'20_partial_results/20_nodes_{n_sample}_{rate}_b{batch_size}_t{time_step}.csv'
        test_results_filepath = f'20_partial_results/20_nodes_test_{n_sample}_{rate}_b{batch_size}_t{time_step}.csv'

        test_results_csv = open(test_results_filepath, 'a')
        test_results_writer = csv.writer(test_results_csv)

        results_csv = open(results_filepath, 'a')
        results_writer = csv.writer(results_csv)

        # load samples, output: numpy array
        input_20 = np.load(f'partial_samples_files/20_nodes_{n_sample}_input.npy')
        output_20 = np.load(f'partial_samples_files/20_nodes_{n_sample}_output.npy')
        threshold = np.load(f'partial_samples_files/20_nodes_{n_sample}_threshold.npy')

        matrix_20 = np.load(f'partial_samples_files/20_nodes_{n_sample}_adj.npy')
        adjacency_20 = torch.from_numpy(matrix_20).type(torch.FloatTensor)
        threshold_tensor = torch.from_numpy(threshold).type(torch.FloatTensor)

        # Global parameters
        number_decimal = 3
        # train_ratio = [0.8, 0.6]
        # validation_ratio = 0.3
        # test_ratio = [0.3]

        # Hyper parameters
        # num_epochs = 10
        # batch_size = 50

        # Start Training:
        input_features, output_feature = adjacency_20.shape  # nodes=20, cascades=3, input_feature=120, output_feature=60

        for training_ratio in train_ratio:
            for testing_ratio in test_ratio:
                # 1. Train, Test, Validation data load
                train_set = PartialObservationData(n_sample, input_20, output_20, training_ratio)
                train_size = train_set.data_size
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=3)

                validation_set = PartialObservationData(n_sample, input_20, output_20, validation_ratio)
                validation_size = validation_set.data_size
                validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False, num_workers=3)

                test_set = PartialObservationData(n_sample, input_20, output_20, testing_ratio)
                test_size = test_set.data_size
                test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=3)

                # 2. Training Loop
                nn_model = MultiCascades3(number_decimal, input_features, output_feature, adjacency_20, 20, 3,
                                          threshold_tensor, time_step)

                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

                n_total_step = len(train_loader)
                n_total_valid_step = len(validation_loader)
                n_total_test_step = len(test_loader)

                train_size_list = [train_size] * (n_total_step * num_epochs)
                print(len(train_size_list))
                test_size_list = [test_size] * (n_total_step * num_epochs)
                valid_size_list = [validation_size] * (n_total_step * num_epochs)

                epoch_list = []
                for i in range(1, num_epochs + 1):
                    each_epoch_list = [i] * n_total_step
                    epoch_list.append(each_epoch_list)
                new_epoch_list = list(np.concatenate(epoch_list).flat)

                batch_list = []
                for i in range(num_epochs):
                    each_batch_list = []
                    for j in range(1, n_total_step + 1):
                        each_batch_list.append(j)
                    batch_list.append(each_batch_list)
                new_batch_list = list(np.concatenate(batch_list).flat)

                start_time = time.time()

                train_loss_list = []
                valid_loss_list = []
                test_loss_list = []
                test_error_list = []

                for epoch in range(num_epochs):
                    print('epoch:', epoch)

                    # Set to training
                    for i, (input_, output_) in enumerate(train_loader):
                        print('train_for_one_batch' + '\n')
                        batch_start_time = time.time()

                        optimizer.zero_grad()
                        y_pred = nn_model(input_)
                        train_loss = criterion(y_pred, output_)
                        train_loss.backward()
                        optimizer.step()

                        train_acc = train_loss.item()

                        print(f'train_loss:{train_acc}')
                        train_loss_list.append(train_acc)
                        valid_acc = 0

                        for j, (input__, output__) in enumerate(validation_loader):
                            y_pred__ = nn_model(input__)
                            valid_loss = criterion(y_pred__, output__)
                            valid_acc += valid_loss.item()
                        valid_ = valid_acc / n_total_valid_step
                        print(f'validation_loss:{valid_}')

                        valid_loss_list.append(valid_)

                    # 3. Testing the model
                    print('test_after_one_epoch' + '\n')
                    test_acc_ = 0
                    predict_error = 0
                    for i, (input___, output___) in enumerate(test_loader):
                        y_pred___ = nn_model(input___)
                        error = 0
                        test_loss = criterion(y_pred___, output___)
                        test_acc_ += test_loss.item()

                    test_acc_ = test_acc_ / n_total_test_step
                    print(f'test_loss:{test_acc_}')

                    test_loss_list.append(test_acc_)
                    predict_error = predict_error / n_total_test_step
                    print(f'predicted_error:{predict_error}')

                    test_error_list.append(predict_error)

                test_epoch_list = []
                test_perepoch_list = []
                for i in range(1, num_epochs+1):
                    test_perepoch_list.append(i)
                    test_epoch_list.append(test_perepoch_list)
                new_test_epoch_list = list(np.concatenate(test_epoch_list).flat)

                for v in range(num_epochs):
                    test_results_writer.writerow([new_test_epoch_list[v], test_loss_list[v]])

                for w in range(n_total_step * num_epochs):
                    results_writer.writerow([train_size_list[w],
                                             valid_size_list[w],
                                             test_size_list[w],
                                             new_epoch_list[w],
                                             new_batch_list[w],
                                             train_loss_list[w],
                                             valid_loss_list[w],
                                             ])


def main_kro(number_samples, learning_rate, rate, train_ratio, validation_ratio, test_ratio, num_epochs, batch_size, time_step_):
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_printoptions(precision=4)

    # number_samples = [5000]
    # learning_rate = 0.01
    # rate = '01'

    for n_sample in number_samples:
        for time_step in time_step_:

            t = time.localtime()
            time_record = time.strftime('%H%M%S', t)

            results_filepath = f'kro_partial_results/{time_record}_kro_{n_sample}_{rate}_b{batch_size}_t{time_step}.csv'

            results_csv = open(results_filepath, 'a')
            results_writer = csv.writer(results_csv)

            test_results_filepath = f'kro_partial_results/{time_record}_kro_test_{n_sample}_{rate}_b{batch_size}_t{time_step}.csv'
            test_results_csv = open(test_results_filepath, 'a')
            test_results_writer = csv.writer(test_results_csv)

            # load samples, output: numpy array
            input_kro = np.load(f'partial_samples_files/kro_{n_sample}_input.npy')
            output_kro = np.load(f'partial_samples_files/kro_{n_sample}_output.npy')
            threshold = np.load(f'partial_samples_files/kro_{n_sample}_threshold.npy')

            matrix_kro = np.load(f'partial_samples_files/kro_{n_sample}_adj.npy')
            adjacency_kro = torch.from_numpy(matrix_kro).type(torch.FloatTensor)
            # print(threshold)
            threshold_tensor = torch.from_numpy(threshold).type(torch.FloatTensor)

            # Global parameters
            number_decimal = 3

            # Start Training:
            input_features, output_feature = adjacency_kro.shape  # nodes=1024, cascades=3

            for training_ratio in train_ratio:
                for testing_ratio in test_ratio:
                    # 1. Train, Test, Validation data load

                    sample_split_list = choose_experiment_data_set(n_sample, validation_ratio, training_ratio,
                                                                   testing_ratio, input_kro, output_kro)
                    train_type = 0
                    valid_type = 1
                    test_type = 2

                    train_set = PartialObservationData(sample_split_list, train_type)
                    train_size = train_set.data_size
                    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=3)

                    validation_set = PartialObservationData(sample_split_list, valid_type)
                    validation_size = validation_set.data_size
                    validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False, num_workers=3)

                    test_set = PartialObservationData(sample_split_list, test_type)
                    test_size = test_set.data_size
                    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=3)

                    # 2. Training Loop
                    nn_model = MultiCascades3(number_decimal, input_features, output_feature, adjacency_kro, 1024, 3,
                                              threshold_tensor, time_step)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

                    decay_rate = 0.9
                    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

                    n_total_step = len(train_loader)
                    n_total_valid_step = len(validation_loader)
                    n_total_test_step = len(test_loader)

                    train_size_list = [train_size] * (n_total_step * num_epochs)
                    # print(len(train_size_list))
                    test_size_list = [test_size] * (n_total_step * num_epochs)
                    valid_size_list = [validation_size] * (n_total_step * num_epochs)

                    epoch_list = []
                    for i in range(1, num_epochs + 1):
                        each_epoch_list = [i] * n_total_step
                        epoch_list.append(each_epoch_list)
                    new_epoch_list = list(np.concatenate(epoch_list).flat)

                    batch_list = []
                    for i in range(num_epochs):
                        each_batch_list = []
                        for j in range(1, n_total_step + 1):
                            each_batch_list.append(j)
                        batch_list.append(each_batch_list)
                    new_batch_list = list(np.concatenate(batch_list).flat)

                    start_time = time.time()

                    train_loss_list = []
                    valid_loss_list = []
                    test_loss_list = []
                    test_error_list = []

                    for epoch in range(num_epochs):
                        print('epoch:', epoch)
                        # print(f'learning_rate:{learning_rate}')

                        # Set to training
                        for i, (input_, output_) in enumerate(train_loader):
                            print('train_for_one_batch' + '\n')
                            batch_start_time = time.time()

                            optimizer.zero_grad()
                            y_pred = nn_model(input_)
                            train_loss = criterion(y_pred, output_)
                            train_loss.backward()
                            optimizer.step()

                            # lr_scheduler.step()
                            # print(f'learning_rate:{lr_scheduler.get_lr()}')

                            train_acc = train_loss.item()

                            print(f'train_loss:{train_acc}')
                            train_loss_list.append(train_acc)
                            valid_acc = 0

                            # print('parameters:')
                            # for name, param in nn_model.named_parameters():
                            #     if param.requires_grad:
                            #         print(name, param.data, param.grad)
                            # print(list(nn_model.parameters()))

                            for j, (input__, output__) in enumerate(validation_loader):
                                y_pred__ = nn_model(input__)
                                valid_loss = criterion(y_pred__, output__)
                                valid_acc += valid_loss.item()
                            valid_ = valid_acc / n_total_valid_step
                            print(f'validation_loss:{valid_}')

                            valid_loss_list.append(valid_)

                        # 3. Testing the model
                        print('test_after_one_epoch' + '\n')
                        test_acc_ = 0
                        predict_error = 0
                        for i, (input___, output___) in enumerate(test_loader):
                            y_pred___ = nn_model(input___)
                            error = 0
                            test_loss = criterion(y_pred___, output___)
                            test_acc_ += test_loss.item()

                            # for sample_idx_in_a_batch in range(batch_size):
                            #     for j in range(output_feature):
                            #         if y_pred___[sample_idx_in_a_batch][j].item() != output___[sample_idx_in_a_batch][j].item():
                            #             error += 1
                            #
                            # test_error = error / (batch_size * output_feature)
                            # predict_error += test_error

                        test_acc_ = test_acc_ / n_total_test_step
                        print(f'test_loss:{test_acc_}')

                        test_loss_list.append(test_acc_)
                        predict_error = predict_error / n_total_test_step
                        print(f'predicted_error:{predict_error}')

                        test_error_list.append(predict_error)

                    # test_results_filepath = f'kro_partial_results/kro_test_{n_sample}_{rate}_b{batch_size}_t{time_step}.csv'
                    # test_results_csv = open(test_results_filepath, 'w')
                    # test_results_writer = csv.writer(test_results_csv)

                    test_epoch_list = []
                    test_perepoch_list = []
                    for i in range(1, num_epochs+1):
                        test_perepoch_list.append(i)
                        test_epoch_list.append(test_perepoch_list)
                    new_test_epoch_list = list(np.concatenate(test_epoch_list).flat)

                    for v in range(num_epochs):
                        test_results_writer.writerow([new_test_epoch_list[v], test_loss_list[v]])

                    for w in range(n_total_step * num_epochs):
                        results_writer.writerow([train_size_list[w],
                                                 valid_size_list[w],
                                                 test_size_list[w],
                                                 new_epoch_list[w],
                                                 new_batch_list[w],
                                                 train_loss_list[w],
                                                 valid_loss_list[w],
                                                 ])
import csv
from full_observation.Full_observation_functions import *


n_nodes = 1024
s_cascade = 3
t_diffusion = 3
t_choose = 3
iteration_step = [1, 2, 5]
number_samples = 1000
num_training = [50, 100, 500]
num_testing = [50, 100, 500]

file = open('../../CLT_influence_function/data/kro_diffusionModel.txt')
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

# number of edges (fixed)
r = len(g.edges)
# set all weights to q decimal
q = 3

g_init = add_attr_on_graph(g, s_cascade, q)
initial_dict_list, sample_set = generate_sample_set('full_samples_files/kro_samples.txt', g_init, number_samples, t_diffusion, s_cascade)
# exit()
with open('kro_results/lr_results.csv', 'w') as lp_csv:
    example_writer = csv.writer(lp_csv, delimiter=',')
    example_writer.writerow(['number_training', 'number_testing', 'number_cascade', 't_iteration', 'train_error', 'test_error'])
    for train_size in num_training:
        print('train_size:', train_size)
        for test_size in num_testing:
            print('test_size:', test_size)
            for t_iteration in iteration_step:
                print('iteration:', t_iteration)
                for each_run in range(5):
                    print('run:', each_run)
                    train_error, test_error = lr_experiments('full_samples_files/kro_samples.txt', g_init, number_samples, train_size, test_size, t_diffusion, t_choose, t_iteration, s_cascade, n_nodes)
                    example_writer.writerow([train_size, test_size, s_cascade, t_iteration, train_error, test_error])



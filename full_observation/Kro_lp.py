import csv
from full_observation.Full_observation_functions import *


n_nodes = 1024
s_cascade = 3

number_samples = 1000
num_training = [10, 50, 100, 500]
num_testing = [10, 50, 100, 500]
# num_training = 500
# num_testing = 1

file = open('../data/Kro_data/kro_diffusionModel.txt')
lines = file.readlines()
edge_list = []
for line in lines:
    li = line.split(' ')
    node_u = int(li[0])
    node_v = int(li[1])
    if node_u < n_nodes+1 and node_v < n_nodes+1:
        edge = (node_u, node_v)
        edge_list.append(edge)
# print(edge_list)


nodes_list = []
for node in range(n_nodes):
    nodes_list.append(node)
g = nx.DiGraph()
g.add_nodes_from(nodes_list)
g.add_edges_from(edge_list)
A = nx.adjacency_matrix(g)
print(A)
# number of edges (fixed)
r = len(g.edges)
# set all weights to q decimal
q = 3


# exit()
g_init = add_attr_on_graph(g, s_cascade, q)

# print('--------------------------------------------------------------------------------------------')
# print('initial_attributes_on_graph')
# print(*g_init.edges.data(), sep='\n')
# print(*g_init.nodes.data(), sep='\n')
# print('--------------------------------------------------------------------------------------------')

# nx.write_gexf(g_init, 'graph_kro.gexf')
initial_dict_list, sample_set = generate_sample_set('full_samples_files/kro_samples.txt', g_init, number_samples, n_nodes, s_cascade)
# g_experiment = nx.read_gexf('graph_kro.gexf', node_type=int)s
# print(g_experiment.nodes.data())
# node_attr = nx.get_node_attributes(g_experiment, 'attr')
# print(node_attr[0])
# exit()

with open('kro_results/lp_results_corrected.csv', 'w') as lp_csv:
    # g_experiments = nx.read_gexf('graph_kro.gexf', node_type=int)
    example_writer = csv.writer(lp_csv, delimiter=',')
    example_writer.writerow(['number_training', 'number_testing', 'number_cascade', 'train_error', 'test_error'])
    for train_size in num_training:
        for test_size in num_testing:
            for each_run in range(5):
                train_error, test_error = lp_results(number_samples, g_init, train_size, test_size, s_cascade, n_nodes)
                example_writer.writerow([train_size, test_size, s_cascade, train_error, test_error])

    # for each_run in range(10):
    #
    #     train_error, test_error = lp_results(number_samples, g_init, num_training, num_testing, s_cascade, n_nodes)
    #     example_writer.writerow([num_training, num_testing, s_cascade, train_error, test_error])


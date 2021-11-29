from full_observation.Full_observation_functions import *
import csv

n_nodes = 1024
s_cascade = 3
t_diffusion = 3
t_choose = 3
number_samples = 1000


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
initial_dict_list, sample_set = generate_sample_set('kro_samples_01.txt', g_init, number_samples, t_diffusion, s_cascade)
guess_sample_size = [10, 50, 100, 500]
with open('kro_results/random_guess.csv', 'w') as random:
    example_writer = csv.writer(random, delimiter=',')
    example_writer.writerow(['sample_size', 'error'])
    for sample_size in guess_sample_size:
        error = random_guess('kro_samples_01.txt', 1000, sample_size, g_init, t_diffusion, t_choose, s_cascade, n_nodes)
        example_writer.writerow([sample_size, error])


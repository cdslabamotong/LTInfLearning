from full_observation.Full_observation_functions import *
from Partial_observation_model_functions import *


number_samples = [1500]
number_cascades = 3
number_decimal = 3


# 20 nodes network:
# for n_sample in number_samples:
#     g_20, threshold_array = load_20_nodes_graph(number_cascades, number_decimal)
#     x_20_samples, y_20_samples = generate_partial_sample_set(g_20, n_sample, 20, number_cascades, threshold_array)
#     print(x_20_samples.shape)
#     print(y_20_samples.shape)
#     np.save(f'partial_samples_files/20_nodes_{n_sample}_input.npy', x_20_samples)
#     np.save(f'partial_samples_files/20_nodes_{n_sample}_output.npy', y_20_samples)
#     np.save(f'partial_samples_files/20_nodes_{n_sample}_threshold.npy', threshold_array)
#
#     # generate 20_matrix:
#     adjacency_matrix_20 = generate_adjacency_matrix(g_20, 20, number_cascades)
#     print(adjacency_matrix_20.shape)
#     # m1, m2, m3, m4 = matrix_in_comparison_part_3_cascade(number_cascades, 20)
#     # np.savez(f'partial_samples_files/20_nodes_{n_sample}', name1=adjacency_matrix_20,
#     #          name2=m1, name3=m2, name4=m3, name5=m4)
#     np.save(f'partial_samples_files/20_nodes_{n_sample}_adj.npy', adjacency_matrix_20)
#
# exit()
# kro network:

for n_sample in number_samples:
    g_kro, threshold_array = load_kro_graph(number_cascades, number_decimal)
    x_kro_samples, y_kro_samples = generate_partial_sample_set(g_kro, n_sample, 1024, number_cascades, threshold_array)

    np.save(f'partial_samples_files/kro_{n_sample}_input.npy', x_kro_samples)
    np.save(f'partial_samples_files/kro_{n_sample}_output.npy', y_kro_samples)
    np.save(f'partial_samples_files/kro_{n_sample}_threshold.npy', threshold_array)

    # generate 20_matrix:
    adjacency_matrix_kro = generate_adjacency_matrix(g_kro, 1024, number_cascades)
    np.save(f'partial_samples_files/kro_{n_sample}_adj.npy', adjacency_matrix_kro)


# for n_sample in number_samples:
#     g_er, threshold_array = load_er_graph(number_cascades, number_decimal)
#     x_er_samples, y_er_samples = generate_partial_sample_set(g_er, n_sample, 5000, number_cascades, threshold_array)
#
#     np.save(f'partial_samples_files/er_{n_sample}_input.npy', x_er_samples)
#     np.save(f'partial_samples_files/er_{n_sample}_output.npy', y_er_samples)
#     np.save(f'partial_samples_files/er_{n_sample}_threshold.npy', threshold_array)
#
#     # generate 20_matrix:
#     adjacency_matrix_kro = generate_adjacency_matrix(g_er, 5000, number_cascades)
#     np.save(f'partial_samples_files/er_{n_sample}_adj.npy', adjacency_matrix_kro)

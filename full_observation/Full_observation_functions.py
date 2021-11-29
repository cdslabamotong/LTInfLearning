import networkx as nx
import re
import json
import numpy as np
import random

from copy import deepcopy
# import cplex
# from docplex.cp.model import CpoModel
# from docplex.mp.model import Model
# import docplex.mp.solution
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
from decimal import *
import pickle


def add_attr_on_graph(graph, number_cascades, decimal_number):
    """
            Parameters
            ----------
            graph : graph
                The graph without attributes
            number_cascades : int
                The number of competitive cascades
            decimal_number : int
                The number of decimals of each parameters, default(3)

            Returns
            ---------
            graph: graph
                The graph with initial attributes
    """

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


def generate_number_seeds(number_nodes):
    """
            Parameters
            ----------
            number_nodes : int
                The number of nodes in graph

            Returns
            ---------
            number_seeds: int
                The number of seeds
    """
    upper_bound = round(number_nodes * 0.5)
    lower_bound = round(number_nodes * 0.1)
    number_seeds = random.randint(lower_bound, upper_bound)
    return number_seeds


def graph_at_step_0(graph, number_cascades):
    """
            Parameters
            ----------
            graph : graph
                The graph with initial attributes
            number_cascades: int
                The number of competitive cascades

            Returns
            ---------
            graph: graph
                The graph with seed set
            init_binary_mode: list
                The nodes status at step 0 in binary mode
    """
    new_graph = graph
    number_nodes = len(new_graph.nodes)
    number_seeds = generate_number_seeds(number_nodes)
    seed_set = random.sample(new_graph.nodes, number_seeds)

    for node in new_graph.nodes:
        if node in seed_set:
            new_graph.nodes[node]['attr']['status'] = random.randint(1, number_cascades)
        else:
            new_graph.nodes[node]['attr']['status'] = 0

    nodes_attr = nx.get_node_attributes(new_graph, 'attr')
    init = [(k, v['status']) for k, v in nodes_attr.items()]
    init_binary_mode = generate_nodes_binary_status(init, number_cascades)
    return new_graph, init_binary_mode


def generate_nodes_binary_status(nodes_status, number_cascades):
    """
            Parameters
            ----------
            nodes_status: list
                The status for each node (if number_cascades = 3): [(1, 1),(2, 3),...,(n, 0)]
            number_cascades: int
                The number of competitive cascades

            Returns
            ---------
            node_status_binary_mode: list
                [(1, 0, 0), (0, 0, 1),...,(0, 0, 0)]
    """
    nodes_status_binary_mode = []
    for instance in nodes_status:
        one_node = []
        for cascade in range(1, number_cascades + 1):
            if instance[1] == cascade:
                one_node.append(1)
            else:
                one_node.append(0)
        nodes_status_binary_mode.append(one_node)
    return nodes_status_binary_mode


def local_diffusion(graph_with_seed, number_cascades):
    """
            Parameters
            ----------
            graph_with_seed : graph
                The graph with seed set
            number_cascades: int
                The number of competitive cascades

            Returns
            ---------
            inter_binary_mode: ls
                The inter status of each node in binary mode
            final_binary_mode: ls
                The final status of each node in binary mode
            stop: int
    """
    nodes_attr = deepcopy(nx.get_node_attributes(graph_with_seed, 'attr'))
    # print('node_status', nodes_attr)
    inter_nodes_attr = deepcopy(nx.get_node_attributes(graph_with_seed, 'attr'))
    final_nodes_attr = deepcopy(nx.get_node_attributes(graph_with_seed, 'attr'))
    edges_attr = deepcopy(nx.get_edge_attributes(graph_with_seed, 'attr'))

    inter = [(k, v['status']) for k, v in inter_nodes_attr.items()]
    inter_binary_mode = generate_nodes_binary_status(inter, number_cascades)
    final = [(k, v['status']) for k, v in final_nodes_attr.items()]
    final_binary_mode = generate_nodes_binary_status(final, number_cascades)
    stop = 0
    for node in graph_with_seed.nodes:
        predecessors = list(graph_with_seed.predecessors(node))
        if len(predecessors) == 0:
            continue
        weight_sum_ls = []
        if nodes_attr[node]['status'] != 0:
            continue

        for cascade in range(number_cascades):
            sum_weight_one_cascade = 0
            for parent_node in predecessors:
                if nodes_attr[parent_node]['status'] == cascade + 1:
                    sum_weight_one_cascade += edges_attr[(parent_node, node)][f'weight_{cascade + 1}']

            if sum_weight_one_cascade >= nodes_attr[node][f'threshold_{cascade + 1}']:
                inter_binary_mode[node][cascade] = 1
            else:
                sum_weight_one_cascade = 0

            weight_sum_ls.append(sum_weight_one_cascade)

        num_candidate_cascades = inter_binary_mode[node].count(1)

        if num_candidate_cascades == 1:
            activated_cascade = inter_binary_mode[node].index(1)
            final_nodes_attr[node]['status'] = activated_cascade + 1
            final_binary_mode[node][activated_cascade] = 1

        if num_candidate_cascades > 1:

            # max_sum = max(weight_sum_ls)
            # max_cascade_index = weight_sum_ls.index(max_sum)

            max_weight, max_index, second_max, second_max_index = find_second_max(weight_sum_ls)

            final_nodes_attr[node]['status'] = max_index + 1
            final_binary_mode[node][max_index] = 1
    # print('final_node_attr:', final_nodes_attr)
    if final_nodes_attr == nx.get_node_attributes(graph_with_seed, 'attr'):
        stop = 1
        # nx.set_node_attributes(graph_with_seed, final_nodes_attr, 'attr')
    else:
        nx.set_node_attributes(graph_with_seed, final_nodes_attr, 'attr')

    return inter_binary_mode, final_binary_mode, stop


def diffusion_process_one_sample(graph_with_seed, number_cascades, num_nodes):
    """
            Parameters
            ----------
            graph_with_seed : graph
                The graph with seed set
            number_cascades: int
                The number of competitive cascades
            num_nodes: int
            Returns
            ---------
            one_sample: list
                one_sample = [[[inter],[final]]*diffusion step]
    """
    one_sample = []

    # for i in range(1, diffusion_step + 1):
    diffusion_step = 0
    for i in range(1, num_nodes + 1):
        one_time = []
        inter, final, stop = local_diffusion(graph_with_seed, number_cascades)
        if stop == 0:
            one_time.append(inter)
            one_time.append(final)
            one_sample.append(one_time)
            diffusion_step += 1
        else:
            one_time.append(inter)
            one_time.append(final)
            one_sample.append(one_time)
            diffusion_step += 1
            break
    return one_sample, diffusion_step


# def diffusion_process_one_sample_01(graph_with_seed, diffusion_step, number_cascades):
#     """
#             Parameters
#             ----------
#             graph_with_seed : graph
#                 The graph with seed set
#             diffusion_step: int
#                 The number of diffusion steps
#             number_cascades: int
#                 The number of competitive cascades
#
#             Returns
#             ---------
#             one_sample: list
#                 one_sample = [[[inter],[final]]*diffusion step]
#     """
#     one_sample = []
#
#     for i in range(1, diffusion_step):
#         one_time = []
#         # print('time_step:', i)
#         inter, final = local_diffusion(graph_with_seed, number_cascades)
#         one_time.append(inter)
#         one_time.append(final)
#         one_sample.append(one_time)
#     return one_sample


def generate_sample_set(filepath, graph, number_samples, number_nodes, number_cascades):
    """
            Parameters
            ----------
            filepath : str
                The file path of sample set
            graph: graph
                The initial graph
            number_samples: int
                The number of generated samples
            number_nodes: int
            number_cascades: int
                The number of competitive cascades

            Returns
            ---------
            initial_dict_list: list
                initial_dict_list = [{}*number_samples]
            sample_set: list
                sample_set = [[[step_0],[step_1],..,[step_t]]*number_samples]
                step_0 = [initial]
                step_t = [inter],[final]
    """

    print('Begin Generating Samples')
    sample_file = open(filepath, 'w')
    initial_dict_list = []
    sample_set = []
    diffusion_step = []
    count = 0
    for sample in range(number_samples):
        sample_file.write(f'sample_{count}' + '\n')
        count += 1
        # print('------------------------------------------------------------------------------------------------------')
        # print(f'sample_{sample}')
        graph_step_0, initial_binary_mode = graph_at_step_0(graph, number_cascades)
        initial_dict = nx.get_node_attributes(graph_step_0, 'attr')
        # print('====================================================================================')
        # print(initial_dict)
        # print('====================================================================================')

        initial_dict_list.append(deepcopy(initial_dict))

        sample_file.write(' '.join([str(elem) for elem in initial_binary_mode]) + '\n')
        one_sample, t_step = diffusion_process_one_sample(graph_step_0, number_cascades, number_nodes)

        sample_set.append(one_sample)
        for index, one_step in enumerate(one_sample):
            sample_file.write(' '.join([str(elem) for elem in one_step]) + '\n')
        sample_file.write(f'{t_step}' + '\n')
    # print('initial_dict_list')
    # print(*initial_dict_list, sep='\n')
    sample_file.close()

    print('End Generating Samples')

    return initial_dict_list, sample_set


def str_to_list_1(string):
    """
            Parameters
            ----------
            string : str
                The str of first line in each sample of sample.txt

            Returns
            ---------
            final_list: lst
    """
    final_list = []
    li = re.findall(r'\[.*?\]', string)
    for ele in li:
        final_list.append(json.loads(ele))
    return final_list


def str_to_list_2(string):
    """
            Parameters
            ----------
            string : str
                The str of rest lines in each sample of sample.txt

            Returns
            ---------
            rst[0]: lst
            rst[1]: lst
    """
    rst = []
    li = re.findall(r'\[\[.*?\]\]', string)
    for ele in li:
        sub_rst = []
        f = re.findall(r'\[.*?\]', ele[1:-1])
        for g in f:
            sub_rst.append(json.loads(g))
        rst.append(sub_rst)
    return rst[0], rst[1]


def find_sample(file_lines, sample_index, num_training, idx):
    """
            Parameters
            ----------
            file_lines : list
            sample_index: int
            num_training: int
            idx: int
            Returns
            ---------
            diffusion_step: int
            first_line_index: int
    """
    # print('idx:', idx)
    if idx + 1 == num_training:
        diffusion_step = int(file_lines[-1])
        # print(diffusion_step)
        first_line_index = len(file_lines) - diffusion_step - 1 - 1
    else:
        search = f'sample_{sample_index + 1}' + '\n'
        # print(sample_index + 1)
        line_number = 0
        t_step_line_index = 0
        for line in file_lines:
            if search in line:
                # print(line)
                t_step_line_index = line_number - 1
            line_number += 1
        # print(t_step_line_index)
        # print(file_lines[t_step_line_index])
        diffusion_step = int(file_lines[t_step_line_index])
        first_line_index = t_step_line_index - (diffusion_step + 1)
    return diffusion_step, first_line_index


def find_sample_list(file_lines, sample_index, num_training, idx):
    """
            Parameters
            ----------
            file_lines : list
            sample_index: int
            num_training: int
            idx: int
            Returns
            ---------
            one_sample: list
    """
    t_diffusion, first_line_index = find_sample(file_lines, sample_index, num_training, idx)
    one_sample = [file_lines[first_line_index]]
    for i in range(t_diffusion):
        one_sample.append(file_lines[first_line_index + i + 1])
    # print(t_diffusion)
    return t_diffusion, one_sample


def choose_samples(filepath, number_samples, graph, number_choose_samples, number_cascades, number_nodes):
    """
    Parameters
    ----------
    filepath : str
        The file path of sample set
    number_samples: int
        The number of generated samples
    graph: graph
        Initial graph
    number_choose_samples: int
        The number of choose samples for training or testing
    number_cascades: int
        The number of competitive cascades
    number_nodes: int
        The number of nodes in graph

    Returns
    ---------
    inter_status: lst
        inter_status = [[[[one_node_status_binary]*num_nodes]*num_choose_step]*num_choose_samples]
    final_status: lst
        final_status = [[[[one_node_status_binary]*num_nodes]*num_choose_step]*num_choose_samples]
    choose_sample_set: lst
    choose_sample_initial_dict_lst: lst
    """

    choose_sample_index_list = random.sample(range(number_samples), number_choose_samples)
    choose_sample_index_list.sort()
    # print('============================================================================================')
    # print('choose_samples_index')
    # print(choose_sample_index_list)

    file = open(filepath, 'r')
    lines = file.readlines()
    inter_status = []
    final_status = []
    choose_sample_initial_dict_lst = []
    choose_samples_set = []
    nodes_attr = nx.get_node_attributes(graph, 'attr')
    diffusion_step = []

    for each_sample_index in choose_sample_index_list:
        for node in range(number_nodes):
            nodes_attr[node]['status'] = 0

        idx = choose_sample_index_list.index(each_sample_index)
        t_diffusion, one_sample = find_sample_list(lines, each_sample_index, number_choose_samples, idx)
        # print('=============================================================================================')
        # print(one_sample)
        diffusion_step.append(t_diffusion)

        single_sample_inter = []
        single_sample_final = []
        single_sample = []

        count = 0
        for line in one_sample:
            if count == 0:
                # print('==============================')
                # print(line)
                # print('=================================')
                # print(line)
                first_line_list = str_to_list_1(line.strip())

                for node in range(number_nodes):
                    ind = 0
                    for each_cascade in first_line_list[node]:
                        ind += 1
                        if each_cascade == 1:
                            nodes_attr[node]['status'] = ind

                choose_sample_initial_dict_lst.append(deepcopy(nodes_attr))

                single_sample_final.append(first_line_list)
                single_sample_inter.append([[0] * number_cascades] * number_nodes)
                count += 1

            else:
                # if t_diffusion == 0:
                #     continue
                one_step = []
                count += 1
                inter, final = str_to_list_2(line.strip())
                single_sample_inter.append(inter)
                single_sample_final.append(final)
                one_step.append(inter)
                one_step.append(final)
                # print('one_step', one_step)
                single_sample.append(deepcopy(one_step))

        inter_status.append(single_sample_inter)
        final_status.append(single_sample_final)
        choose_samples_set.append(deepcopy(single_sample))
    return inter_status, final_status, choose_samples_set, choose_sample_initial_dict_lst, diffusion_step, choose_sample_index_list


def check_whether_activated(lst):
    for ele in lst:
        if ele == 1:
            return True


def find_second_max(lst):
    if lst[0] == lst[1]:
        mx = lst[0]
        mx_index = 0
        second_max = lst[1]
        second_max_index = 1
    else:
        mx = max(lst[0], lst[1])
        mx_index = (lst[0], lst[1]).index(mx)
        second_max = min(lst[0], lst[1])
        second_max_index = (lst[0], lst[1]).index(second_max)

    n = len(lst)
    for i in range(2, n):
        if lst[i] > mx:
            second_max = mx
            second_max_index = mx_index
            mx = lst[i]
            mx_index = i
        elif lst[i] == mx:
            if mx == second_max:
                break
            else:
                second_max = lst[i]
                second_max_index = i
        elif second_max < lst[i] != mx:
            second_max = lst[i]
            second_max_index = i
    return mx, mx_index, second_max, second_max_index


def generate_lp_model(graph_init, inter_status, final_status, number_choose_samples, diffusion_step, number_cascades, number_nodes):
    """
    Parameters
    ----------
    graph_init: graph
    inter_status: list
    final_status: list
    number_choose_samples: int
        The number of choose samples for training or testing
    diffusion_step: list
    number_cascades: int
        The number of competitive cascades
    number_nodes: int
        The number of nodes in graph

    Returns
    ---------
    weight_matrix: np.matrix
    threshold_matrix: np.matrix
    """
    lp_record = open('kro_results/lp_results_record.txt', 'w')
    # create LP model
    mdl = Model()
    weight = {(j, i, k): mdl.integer_var(name='weight_{0}_{1}^{2}'.format(j, i, k), lb=0, ub=999) for i in range(number_nodes)
              for j in list(graph_init.predecessors(i)) for k in range(number_cascades)}

    threshold = {(i, j): mdl.integer_var(name='threshold_{0}^{1}'.format(i, j), lb=0, ub=999) for i in range(number_nodes)
                 for j in range(number_cascades)}


    for sample_count in range(number_choose_samples):
        t_diffusion = diffusion_step[sample_count]
        for step_count in range(t_diffusion):
            for node_count in range(number_nodes):
                if check_whether_activated(final_status[sample_count][step_count][node_count]):
                    continue
                if len(list(graph_init.predecessors(node_count))) == 0:
                    continue
                one_node_sum_weight_list = []
                one_node_sum_weight_ground_list = []
                max_sum_weight = 0
                max_sum_weight_ground = 0
                for cascade_count in range(number_cascades):
                    # if final_status[sample_count][step_count][node_count][cascade_count] == 1:
                    #     continue
                    sum_weight = 0
                    ground_sum_weight = 0
                    for parent_node in list(graph_init.predecessors(node_count)):
                        if final_status[sample_count][step_count][parent_node][cascade_count] == 1:

                            sum_weight += weight[parent_node, node_count, cascade_count]/1000
                            ground_sum_weight += graph_init.edges[parent_node, node_count]['attr'][f'weight_{cascade_count + 1}']
                    one_node_sum_weight_list.append(sum_weight)
                    one_node_sum_weight_ground_list.append(ground_sum_weight)
                    # print('sample:', sample_count, 'step:', step_count, 'cascade:', cascade_count)
                    # print('node:', node_count, 'weight_sum_list:', one_node_sum_weight_ground_list)

                    if inter_status[sample_count][step_count + 1][node_count][cascade_count] == 1:
                        constrains_1 = (2 * inter_status[sample_count][step_count + 1][node_count][cascade_count] - 1) * \
                                   (sum_weight - threshold[node_count, cascade_count]/1000)
                        ground_truth_1 = (2 * inter_status[sample_count][step_count + 1][node_count][cascade_count] - 1) * \
                                         (ground_sum_weight - graph_init.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                        # if ground_truth_1 >= 0:
                        #     print('YES')
                        if ground_truth_1 < 0:
                            print('NO')
                            print('ground_sum_weight:', ground_sum_weight)
                            print('weight_sum - threshold:', ground_sum_weight - graph_init.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            print('con_1: sample, step, node, cascade:', sample_count, step_count + 1, node_count, cascade_count)
                            print('inter:', inter_status[sample_count][step_count + 1][node_count][cascade_count])
                            # exit()

                        mdl.add(constrains_1 >= 0)
                        # constrains_1 = 0
                    else:
                        # print(2 * inter_status[sample_count][step_count+1][node_count][cascade_count] - 1)
                        constrains_2 = (2 * inter_status[sample_count][step_count + 1][node_count][cascade_count] - 1) * \
                                   (sum_weight - threshold[node_count, cascade_count]/1000)
                        ground_truth_2 = (2 * inter_status[sample_count][step_count + 1][node_count][cascade_count] - 1)\
                                         * (ground_sum_weight - graph_init.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                        # if ground_truth_2 >= 0.00001:
                        #     print('YES')
                        if ground_truth_2 < 0.0001:
                            print(2 * inter_status[sample_count][step_count + 1][node_count][cascade_count] - 1)
                            print('NO')
                            print('ground_sum_weight:', ground_sum_weight)
                            print('threshold:', graph_init.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            print('weight_sum - threshold:', ground_sum_weight - graph_init.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            print('con_2: sample, step, node, cascade:', sample_count, step_count + 1, node_count, cascade_count)
                            print('inter:', inter_status[sample_count][step_count + 1][node_count][cascade_count])
                            # exit()

                        # mdl.add(x_0[sample_count, step_count] != 0)
                        # mdl.add(constrains_2 - x_0[sample_count, step_count] >= 0)
                        mdl.add(constrains_2 >= 0.0001)

                        # constrains_2 = 0
                    # mdl.add(constrains_1 <= 1)

                # b = []

                if inter_status[sample_count][step_count + 1][node_count] == [0] * number_cascades:
                    continue

                if inter_status[sample_count][step_count + 1][node_count].count(1) == 1:
                    continue

                # g_max_weight_cascade_ground = int
                for each_cascade in range(number_cascades):
                    if inter_status[sample_count][step_count + 1][node_count][each_cascade] == 0:
                        one_node_sum_weight_list[each_cascade] = 0
                        one_node_sum_weight_ground_list[each_cascade] = 0

                # lp_record = open('lp_results_record.txt', 'w')

                if inter_status[sample_count][step_count + 1][node_count].count(1) > 1:
                    lp_record.write(f'{sample_count},' + f'{step_count},' + f'{node_count}' + '\n')
                    lp_record.write(f'{one_node_sum_weight_ground_list}' + '\n')

                g_max_weight, g_max_weight_cascade, g_second_weight, g_second_weight_cascade = find_second_max(one_node_sum_weight_ground_list)

                lp_record.write(f'{g_max_weight},' + f'{g_max_weight_cascade},' + f'{g_second_weight},' + f'{g_second_weight_cascade},' + '\n')
                lp_record.write('=====================================================================================')
                # lp_record.close()

                # diff = one_node_sum_weight_list[g_max_weight_cascade] - one_node_sum_weight_list[g_second_weight_cascade]
                diff_ground = one_node_sum_weight_ground_list[g_max_weight_cascade] - one_node_sum_weight_ground_list[g_second_weight_cascade]

                for cascade in range(number_cascades):
                    diff = one_node_sum_weight_list[g_max_weight_cascade] - one_node_sum_weight_list[cascade]
                    if cascade == g_second_weight_cascade:
                        if one_node_sum_weight_ground_list[g_max_weight_cascade] == one_node_sum_weight_ground_list[g_second_weight_cascade]:
                            mdl.add(diff == 0)
                        else:
                            mdl.add(diff >= 0.0001)
                    elif cascade == g_max_weight_cascade:
                        mdl.add(diff == 0)
                    else:
                        mdl.add(diff >= 0.0001)

                if diff_ground < 0:
                    print('NO')
                    print(one_node_sum_weight_ground_list)
                    print(one_node_sum_weight_ground_list[g_max_weight_cascade])
                    print(one_node_sum_weight_ground_list[g_second_weight_cascade])
                # if g_max_weight == g_second_weight:
                #     mdl.add(diff == 0)
                #     if diff_ground != 0:
                #         print('NO')
                # else:
                #     mdl.add(diff >= 0.00001)
                #     if diff_ground < 0.00001:
                #         print('NO')


                    # # if diff_ground >= 0:
                    # #     print('YES')
                    # if diff_ground < 0:
                    #     print('NO')
                    #     print('cascade:', each_cascade, 'sum_weight:', one_node_sum_weight_ground_list[each_cascade])
                    #     print('max_cascade:', max_cascade_num)
                    #     print('max_sum_weight_ground:', max_sum_weight_ground, 'ele:', one_node_sum_weight_ground_list[each_cascade])
                    #     print('con_3: sample, step, node, max_cascade_num:', sample_count, step_count + 1, node_count, max_cascade_num)
                    #     exit()
                    #
                    # mdl.add(diff >= 0)
                    # b = []

    # for j in range(number_nodes):
    #     for i in range(number_nodes):
    #         sum_w = 0
    #         for k in range(number_cascades):
    #             sum_w += weight[(j, i, k)]/1000
    #         mdl.add(sum_w <= 1)




    mdl_sol = mdl.solve()
    # mdl_1 = mdl.populate()
    print('Solution_details:')
    print(mdl.solve_details)
    # print('Solution')
    # print(mdl_sol)
    # print('check_valid')
    # print()
    # mdl_sol.display()

    a = np.array([[[0]*number_cascades]*number_nodes]*number_nodes, dtype=float)
    weight_matrix = a.astype(np.float32)

    b = np.array([[0]*number_cascades]*number_nodes, dtype=float)
    threshold_matrix = b.astype(np.float32)

    for i in range(number_nodes):
        for j in list(graph_init.predecessors(i)):
            for k in range(number_cascades):
                # print(f'w_{j,i}^{k}:', weight[j, i, k])
                weight_matrix[j][i][k] = weight[j, i, k]
                threshold_matrix[i][k] = threshold[i, k]
    lp_record.close()
    return weight_matrix, threshold_matrix


def generate_graph_with_predict_parameters(graph_init, number_nodes, number_cascades, weight_matrix, threshold_matrix):
    """
    Parameters
    ----------
    graph_init: graph
    number_nodes: int
        The number of nodes in graph
    number_cascades: int
        The number of competitive cascades
    weight_matrix: np.matrix
    threshold_matrix: np.matrix

    Returns
    ---------
    new_graph: graph with predicted parameters
    """
    # print('--------------------------------------------------')
    # print(*graph_init.nodes.data(), sep='\n')
    new_graph = deepcopy(graph_init)
    edge_new_attr = nx.get_edge_attributes(new_graph, 'attr')
    node_new_attr = nx.get_node_attributes(new_graph, 'attr')
    for node in range(number_nodes):
        predecessors = list(graph_init.predecessors(node))
        for parent in predecessors:
            for cascade in range(number_cascades):
                edge_new_attr[(parent, node)][f'weight_{cascade+1}'] = weight_matrix[parent][node][cascade]/1000
    nx.set_edge_attributes(new_graph, edge_new_attr, 'attr')
    for node in range(number_nodes):
        for cascade in range(number_cascades):
            node_new_attr[node][f'threshold_{cascade+1}'] = threshold_matrix[node][cascade]/1000
    nx.set_node_attributes(new_graph, node_new_attr, 'attr')
    return new_graph


def check_error_lp(graph_init, graph_with_predict_parameter, initial_dict_list, diffusion_list, number_cascades,
                   num_train, num_nodes, choose_samples_set):
    """
    Parameters
    ----------
    graph_init: graph
    graph_with_predict_parameter: graph
    initial_dict_list: list
    diffusion_list: list
    number_cascades: int
        The number of competitive cascades
    num_train: int
    num_nodes: int
    choose_samples_set: list

    Returns
    ---------
    error: float
    """

    # print(graph_init.nodes.data())
    # print(graph_with_predict_parameter.nodes.data())
    # graph_predicted_initial = deepcopy(graph_with_predict_parameter)
    predict_samples = []
    # predicted_samples = []
    node_attr = nx.get_node_attributes(graph_with_predict_parameter, 'attr')
    # edge_attr = nx.get_edge_attributes(graph_with_predict_parameter,)
    count = 0
    for initial_dict in initial_dict_list:

        for node in range(num_nodes):
            node_attr[node]['status'] = initial_dict[node]['status']

        nx.set_node_attributes(graph_with_predict_parameter, node_attr, 'attr')
        # print(f'sample{count}')
        # print(graph_with_predict_parameter.nodes.data())
        # graph_predicted_initial = deepcopy(graph_with_predict_parameter)

        t_diffusion = diffusion_list[count]
        one_predict_sample, t_predicted = diffusion_process_one_sample(graph_with_predict_parameter, number_cascades, num_nodes)
        # print(t_diffusion, t_predicted)
        # # print(one_predict_sample)
        # print('=============================================================================')
        predict_samples.append(one_predict_sample)
        count += 1

    # samples_matrix = np.array(choose_samples_set)
    # predict_samples_matrix = np.array(predict_samples)
    # print(*predict_samples, sep='\n')
    # print('========================================================================')
    # print(*choose_samples_set, sep='\n')


    count = 0
    for sample_count in range(num_train):
        # print('sample:', sample_count)
        # t = diffusion_list[sample_count]
        for each_node in range(num_nodes):
            # print('node:', each_node)
            # print(len(choose_samples_set[sample_count]))
            # print(len(predict_samples[sample_count]))
            array_1 = choose_samples_set[sample_count][-1][-1][each_node]
            # print('ground:', array_1)
            array_2 = predict_samples[sample_count][-1][-1][each_node]
            # print('predicted:', array_2)

            if not np.array_equal(array_1, array_2):
                print(f'initial:')
                print(initial_dict_list[sample_count][each_node])
                print(f'{sample_count},{each_node}')
                print('ground_sample:')
                print(*choose_samples_set[sample_count], sep='\n')
                print('predict_sample:')
                print(*predict_samples[sample_count], sep='\n')
                print(f'ground_truth', array_1)
                print(f'predict', array_2)
                print('predicted_threshold:', graph_with_predict_parameter.nodes[each_node]['attr'])
                print('ground_threshold:', graph_init.nodes[each_node]['attr'])
                in_neighbor = list(graph_with_predict_parameter.predecessors(each_node))
                for node in in_neighbor:
                    print('in_neighbor:', node)
                    print('in_neighbor status:', initial_dict_list[sample_count][node]['status'])
                    print('node_pair:', (node, each_node))
                    print('predicted weight:', graph_with_predict_parameter.edges[node, each_node]['attr'])
                    print('ground weight:', graph_init.edges[node, each_node]['attr'])
                count += 1
                print('=============================================================================================')

    error = count / (num_train * num_nodes)
    return error


def check_constrains(graph_with_predicted_para, inter_status, final_status, number_choose_samples, diffusion_list, number_nodes, number_cascades):
    for sample_count in range(number_choose_samples):
        # print('sample:', sample_count)
        t_diffusion = diffusion_list[sample_count]
        for step_count in range(t_diffusion):
            # print('step:', step_count)
            for node_count in range(number_nodes):
                # print('node:', node_count)
                if check_whether_activated(final_status[sample_count][step_count][node_count]):
                    continue
                if len(list(graph_with_predicted_para.predecessors(node_count))) == 0:
                    continue
                one_node_sum_weight_list = []
                max_sum_weight = 0
                for cascade_count in range(number_cascades):
                    sum_weight = 0
                    for parent_node in list(graph_with_predicted_para.predecessors(node_count)):
                        if final_status[sample_count][step_count][parent_node][cascade_count] == 1:
                            sum_weight += graph_with_predicted_para.edges[parent_node, node_count]['attr'][f'weight_{cascade_count + 1}']

                    one_node_sum_weight_list.append(sum_weight)

                    if inter_status[sample_count][step_count + 1][node_count][cascade_count] == 1:
                        con_1 = (2 * 1 - 1) * (sum_weight - graph_with_predicted_para.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                        # if con_1 >= 0:
                        #     print('YES')
                            # print('con_1:', con_1)
                            # print('sum_weight:', sum_weight)
                            # print('threshold:', graph_with_predicted_para.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            # print('con_1: sample, step, node, cascade:', sample_count, step_count + 1, node_count, cascade_count)
                            # print('inter:', inter_status[sample_count][step_count + 1][node_count][cascade_count])
                        if con_1 < 0:
                            print('NO')
                            print('sum_weight:', sum_weight)
                            print('weight_sum - threshold:', sum_weight - graph_with_predicted_para.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            print('con_1: sample, step, node, cascade:', sample_count, step_count + 1, node_count, cascade_count)
                            print('inter:', inter_status[sample_count][step_count + 1][node_count][cascade_count])
                            # exit()
                    else:
                        con_2 = (2 * 0 - 1) * (sum_weight - graph_with_predicted_para.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])

                        # if con_2 >= 0.00001:
                        #     print('YES')
                            # print('con_2:', con_2)
                            # print('sum_weight:', sum_weight)
                            # print('threshold:', graph_with_predicted_para.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            # print('con_2: sample, step, node, cascade:', sample_count, step_count + 1, node_count, cascade_count)
                            # print('inter:', inter_status[sample_count][step_count + 1][node_count][cascade_count])

                        if con_2 < 0.00001:
                            print('NO')
                            print('sum_weight:', sum_weight)
                            print('weight_sum - threshold:', sum_weight - graph_with_predicted_para.nodes[node_count]['attr'][f'threshold_{cascade_count + 1}'])
                            print('con_2: sample, step, node, cascade:', sample_count, step_count + 1, node_count,
                                  cascade_count)
                            print('inter:', inter_status[sample_count][step_count + 1][node_count][cascade_count])
                            # exit()

                # if inter_status[sample_count][step_count + 1][node_count] == [0] * number_cascades:
                #     continue
                #
                # elif inter_status[sample_count][step_count + 1][node_count].count(1) == 1:
                #     continue
                #
                # max_cascade_num = 0
                #
                # for count, ele in enumerate(one_node_sum_weight_list):
                #     # print('cascade:', count, 'sum_weight:', ele)
                #     if final_status[sample_count][step_count + 1][node_count][count] == 1:
                #         max_sum_weight = ele
                #         max_cascade_num = count
                #
                # for each_cascade in range(number_cascades):
                #     if inter_status[sample_count][step_count + 1][node_count][each_cascade] == 0:
                #         one_node_sum_weight_list[each_cascade] = 0
                #
                #     diff = one_node_sum_weight_list[max_cascade_num] - one_node_sum_weight_list[each_cascade]
                #
                #     # if diff >= 0:
                #     #     print('YES')
                #     if diff < 0:
                #         print('NO')
                #         print('cascade:', each_cascade, 'sum_weight:', one_node_sum_weight_list[each_cascade])
                #         print('max_cascade:', max_cascade_num)
                #         print('max_sum_weight_ground:', max_sum_weight, 'ele:', one_node_sum_weight_list[each_cascade])
                #         print('con_3: sample, step, node, max_cascade_num:', sample_count, step_count + 1, node_count, max_cascade_num)
                #         exit()
                if inter_status[sample_count][step_count + 1][node_count] == [0] * number_cascades:
                    continue

                if inter_status[sample_count][step_count + 1][node_count].count(1) == 1:
                    continue

                g_max_weight_cascade = int
                for each_cascade in range(number_cascades):
                    if inter_status[sample_count][step_count + 1][node_count][each_cascade] == 0:
                        one_node_sum_weight_list[each_cascade] = 0
                    if final_status[sample_count][step_count + 1][node_count][each_cascade] == 1:
                        g_max_weight_cascade = each_cascade


                # g_max_weight, g_max_weight_cascade, g_second_weight, g_second_weight_cascade = find_second_max(one_node_sum_weight_list)
                # print(sample_count, step_count, node_count)
                # print(g_max_weight, g_max_weight_cascade, g_second_weight, g_second_weight_cascade)
                for i in range(number_cascades):
                    diff = one_node_sum_weight_list[g_max_weight_cascade] - one_node_sum_weight_list[i]

                    if diff < 0:
                        print('NO')


def lp_results(num_total_samples, graph_init, num_train, num_test, num_cascade, num_nodes):
    print('choose_training_sample_randomly')

    inter, final, choose_sample, initial_dict_set, t_list, choose_index = choose_samples(
        'full_samples_files/kro_samples.txt', num_total_samples, graph_init, num_train, num_cascade, num_nodes)

    # count = 0
    # for ins in choose_sample:
    #     print(f'sample{count}')
    #     # print(initial_dict_set[count])
    #     print(*ins, sep='\n')
    #     index = choose_index[count]
    #     print('index_in_choose_samples:', count, 'index_in_samples:', index)
    #     #
    #     #     if initial_dict_set[count] == initial_dict_list[index]:
    #     #         print('YES')
    #     #     else:
    #     #         print('NO')
    #     count += 1

    e, f = generate_lp_model(graph_init, inter, final, num_train, t_list, num_cascade, num_nodes)
    print('generate graph with predicted parameters')
    graph_predicted = generate_graph_with_predict_parameters(graph_init, num_nodes, num_cascade, e, f)
    print('check training error')
    error_training = check_error_lp(graph_init, graph_predicted, initial_dict_set, t_list, num_cascade, num_train, num_nodes, choose_sample)
    print('check constrains')
    check_constrains(graph_predicted, inter, final, num_train, t_list, num_nodes, num_cascade)
    print(error_training)
    if error_training != 0:
        exit()
    print('=======================================================================================================================================================')
    print('choose_testing_sample_randomly')
    inter_1, final_1, choose_sample_set_1, initial_dict_list_1, t_list_1, choose_index_1 = choose_samples(
        'full_samples_files/kro_samples.txt', num_total_samples, graph_init, num_test, num_cascade, num_nodes)
    print('check testing error')
    error_testing = check_error_lp(graph_init, graph_predicted, initial_dict_list_1, t_list_1, num_cascade, num_test, num_nodes, choose_sample_set_1)
    print(error_testing)
    return error_training, error_testing


def binary_mode_to_cascade_mode(binary_list, num_cascade):
    """
    Parameters
    ----------
    binary_list: list
    num_cascade: int
    Returns
    ---------
    cascade_mode: int
    """
    cascade_mode = 0
    if binary_list == [0] * num_cascade:
        cascade_mode = 0
    count = 1
    for ele in binary_list:
        if ele == 1:
            cascade_mode = count
        count += 1
    return cascade_mode


def generate_lr_format_samples(final_status, num_choose_samples, step_list, num_nodes, num_cascades):
    """
    Parameters
    ----------
    final_status: list
    num_choose_samples: int
    step_list: list
    num_nodes: int
    num_cascades: int
    Returns
    ---------
    lr_format_samples
    """
    lr_format_samples = []
    for sample_count in range(num_choose_samples):
        each_sample = []
        step = step_list[sample_count]
        for step_count in range(step):
            each_step = []
            for node_count in range(num_nodes):
                node_status = binary_mode_to_cascade_mode(final_status[sample_count][step_count][node_count], num_cascades)
                each_step.append(node_status)
            each_sample.append(each_step)
        lr_format_samples.append(each_sample)

    return lr_format_samples


def generate_it_ot_lr(lr_format_samples, graph, num_samples, step_list, num_nodes):
    """
    Parameters
    ----------
    lr_format_samples: list
    graph: graph
    num_samples: int
    step_list: list
    num_nodes: int
    Returns
    ---------
    it: dict
    ot: dict
    """
    it = {}
    ot = {}
    for node in range(num_nodes):
        one_node_it = []
        one_node_ot = []
        in_neighbors = list(graph.predecessors(node))
        if len(in_neighbors) == 0:
            continue
        for sample_count in range(num_samples):
            step = step_list[sample_count]
            for step_count in range(1, step):
                one_node_one_step_it = [lr_format_samples[sample_count][step_count - 1][i] for i in in_neighbors]
                one_node_one_step_ot = lr_format_samples[sample_count][step_count][node]
                one_node_it.append(one_node_one_step_it)
                one_node_ot.append(one_node_one_step_ot)
        it[node] = one_node_it
        ot[node] = one_node_ot
    # print(it)
    # print(ot)
    return it, ot


# def reshape_one_node_it_ot(it_one_node, ot_one_node, num_samples, num_step):
#     """
#     Parameters
#     ----------
#     it_one_node: list
#     ot_one_node: list
#     num_samples: int
#     num_step: int
#     Returns
#     ---------
#     it
#     ot
#     """
#     it = np.reshape(np.array(it_one_node), (num_samples * num_step, -1))
#     ot = np.reshape(np.array(ot_one_node), num_samples * num_step)
#
#     return it, ot


def lr_model(sample_file, num_total_samples, graph_init, num_train, num_cascade, num_nodes):
    """
    Parameters
    ----------
    sample_file: str
    num_total_samples: int
    graph_init: graph
    num_train: int
    num_cascade: int
    num_nodes: int
    Returns
    ---------
    final_status: list
    """

    a, final_status, c, d, t_list, e = choose_samples(sample_file, num_total_samples, graph_init, num_train, num_cascade, num_nodes)

    lr_format_samples = generate_lr_format_samples(final_status, num_train, t_list, num_nodes, num_cascade)
    # print(lr_format_samples)
    lr_it, ground_ot = generate_it_ot_lr(lr_format_samples, graph_init, num_train, t_list, num_nodes)

    count = 0

    LR_model = []
    for node in range(num_nodes):
        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            count += 1
            continue
        model = LogisticRegression(max_iter=10000)
        arr_it = np.array(lr_it[node])
        arr_ot = np.array(ground_ot[node])

        # print('=================================================================================================')
        # print('input:', lr_it[node])

        model.fit(arr_it, arr_ot)
        LR_model.append(model)
        for each_model in LR_model:
            pickle.dump(each_model, open(f'lr_models/{node}_lr_model.sav', 'wb'))

    return final_status, t_list


def svm_model(sample_file, num_total_samples, graph_init, num_train, num_cascade, num_nodes):
    """
    Parameters
    ----------
    sample_file: str
    num_total_samples: int
    graph_init: graph
    num_train: int
    num_cascade: int
    num_nodes: int
    Returns
    ---------
    final_status: list
    """

    a, final_status, b, c, t_list, e = choose_samples(sample_file, num_total_samples, graph_init, num_train, num_cascade, num_nodes)

    lr_format_samples = generate_lr_format_samples(final_status, num_train, t_list, num_nodes, num_cascade)

    lr_it, ground_ot = generate_it_ot_lr(lr_format_samples, graph_init, num_train, t_list, num_nodes)

    count = 0

    SVM_model = []
    for node in range(num_nodes):
        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            count += 1
            continue
        model = OneVsOneClassifier(SVC())
        arr_it = np.array(lr_it[node])
        arr_ot = np.array(ground_ot[node])

        # print('=================================================================================================')
        # print('input:', lr_it[node])

        model.fit(arr_it, arr_ot)
        SVM_model.append(model)
        for each_model in SVM_model:
            pickle.dump(each_model, open(f'svm_models/{node}_lr_model.sav', 'wb'))

    return final_status, t_list


def svm_1_iter_model(sample_file, num_total_samples, graph_init, num_train, num_test, num_cascade, num_nodes):
    """
    Parameters
    ----------
    sample_file: str
    num_total_samples: int
    graph_init: graph
    num_train: int
    num_test: int
    num_cascade: int
    num_nodes: int
    Returns
    ---------
    train_final_status: list
    """

    a, train_final_status, c, d, train_t_list, e = choose_samples(sample_file, num_total_samples, graph_init, num_train, num_cascade, num_nodes)
    a_1, test_final_status, c_1, d_1, test_t_list, e_1 = choose_samples(sample_file, num_total_samples, graph_init, num_test, num_cascade, num_nodes)

    train_sum_step = 0
    test_sum_step = 0
    for i in train_t_list:
        train_sum_step += i

    for j in test_t_list:
        test_sum_step += j

    train_format_samples = generate_lr_format_samples(train_final_status, num_train, train_t_list, num_nodes, num_cascade)
    test_format_samples = generate_lr_format_samples(test_final_status, num_test, test_t_list, num_nodes, num_cascade)

    train_it, train_ground_ot = generate_it_ot_lr(train_format_samples, graph_init, num_train, train_t_list, num_nodes)
    test_it, test_ground_ot = generate_it_ot_lr(test_format_samples, graph_init, num_test, test_t_list, num_nodes)

    count = 0

    train_error_whole_sample = 0
    test_error_whole_sample = 0

    for node in range(num_nodes):
        train_error_one_node = 0
        test_error_one_node = 0

        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            count += 1
            continue
        model = OneVsOneClassifier(SVC())
        train_arr_it = np.array(train_it[node])
        train_ground_arr_ot = np.array(train_ground_ot[node])

        test_arr_it = np.array(test_it[node])
        test_ground_arr_ot = np.array(test_ground_ot[node])

        model.fit(train_arr_it, train_ground_arr_ot)

        train_predict_ot = model.predict(train_arr_it)
        test_predict_ot = model.predict(test_arr_it)

        train_error_num = 0
        for idx, y_train in np.ndenumerate(train_predict_ot):
            if y_train != train_ground_arr_ot[idx]:
                train_error_num += 1

        train_error_one_node = train_error_num/train_sum_step
        train_error_whole_sample += train_error_one_node

        test_error_num = 0
        for idx_1, y_test in np.ndenumerate(test_predict_ot):
            if y_test != test_ground_arr_ot[idx_1]:
                test_error_num += 1
        test_error_one_node = test_error_num/test_sum_step
        test_error_whole_sample += test_error_one_node

    train_error = train_error_whole_sample/(num_nodes - count)
    test_error = test_error_whole_sample/(num_nodes - count)

    return train_error, test_error


def samples_for_iteration(node_status, num_nodes, num_train, step_list, num_cascade):

    """
    Parameters
    ----------
    node_status: list
    num_nodes: int
    num_train: int
    step_list: list
    num_cascade: int
    Returns
    ---------
    node_status_samples_it: list
    node_status_samples_ot: list
    """

    # For each node, transfer binary mode to real value
    lr_format_sample = generate_lr_format_samples(node_status, num_train, step_list, num_nodes, num_cascade)
    # print(lr_format_sample)

    node_status_samples_it = []
    node_status_samples_ot = []
    for each_sample in lr_format_sample:
        node_status_samples_it.append(each_sample[0])
        node_status_samples_ot.append(each_sample[-1])
    # return initial status and final status for whole samples
    return node_status_samples_it, node_status_samples_ot


def samples_each_node_for_iteration(it_before, ot_before, num_nodes, graph, num_train):
    """
    Parameters
    ----------
    it_before: list
    initial status for whole samples
    ot_before: list
    final status for whole samples
    graph: graph
    num_nodes: int
    num_train: int
    Returns
    ---------
    it: dict
    ot: dict
    """
    it = {}
    ot = {}

    for node in range(num_nodes):
        in_neighbors = list(graph.predecessors(node))
        if len(in_neighbors) == 0:
            continue
        one_node_it = []
        one_node_ot = []
        for sample_count in range(num_train):

            one_node_it.append([it_before[sample_count][i] for i in in_neighbors])
            one_node_ot.append(ot_before[sample_count][node])

        it[node] = one_node_it
        ot[node] = one_node_ot
    # return two dict it and ot index by each node
    return it, ot


def lr_first_iteration(node_status, num_nodes, graph_init, num_train, step_list, num_cascade):
    """
    Parameters
    ----------
    node_status: list
    graph_init: graph
    num_nodes: int
    num_train: int
    step_list: list
    num_cascade: int
    Returns
    ---------
    it_before: list
    ot_before: list
    """

    # generate two list
    it_before, ot_before_ground = samples_for_iteration(node_status, num_nodes, num_train, step_list, num_cascade)

    # generate two dict
    lr_it, ground_lr_ot = samples_each_node_for_iteration(it_before, ot_before_ground, num_nodes, graph_init, num_train)

    for node in range(num_nodes):

        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            continue

        arr_it = np.array(lr_it[node])

        model = pickle.load(open(f'lr_models/{node}_lr_model.sav', 'rb'))

        ot_predict = model.predict(arr_it)
        ot_predict.tolist()

        # update the value in the initial list
        for sample_count in range(num_train):
            it_before[sample_count][node] = ot_predict[sample_count]

    # return two list: node status for next step and final status
    return it_before, ot_before_ground


def svm_first_iteration(node_status, num_nodes, graph_init, num_train, step_list, num_cascade):
    """
    Parameters
    ----------
    node_status: list
    graph_init: graph
    num_nodes: int
    num_train: int
    step_list: list
    num_cascade: int
    Returns
    ---------
    it_before: list
    ot_before: list
    """

    # generate two list
    it_before, ot_before_ground = samples_for_iteration(node_status, num_nodes, num_train, step_list, num_cascade)

    # generate two dict
    lr_it, ground_lr_ot = samples_each_node_for_iteration(it_before, ot_before_ground, num_nodes, graph_init, num_train)

    for node in range(num_nodes):

        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            continue

        arr_it = np.array(lr_it[node])

        model = pickle.load(open(f'svm_models/{node}_lr_model.sav', 'rb'))

        ot_predict = model.predict(arr_it)
        ot_predict.tolist()

        # update the value in the initial list
        for sample_count in range(num_train):
            it_before[sample_count][node] = ot_predict[sample_count]

    # return two list: node status for next step and final status
    return it_before, ot_before_ground


def lr_iteration(it_before, ot_before_ground, num_nodes, graph_init, num_train):
    """
    Parameters
    ----------
    it_before: list
    ot_before_ground: list
    graph_init: graph
    num_nodes: int
    num_train: int
    Returns
    ---------
    it_before: list
    """
    lr_it = samples_each_node_for_iteration(it_before, ot_before_ground, num_nodes, graph_init, num_train)[0]

    for node in range(num_nodes):

        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            continue

        arr_it = np.array(lr_it[node])
        # print(arr_it)
        # print(arr_it)
        model = pickle.load(open(f'lr_models/{node}_lr_model.sav', 'rb'))

        ot_predict = model.predict(arr_it)

        for sample_count in range(num_train):
            it_before[sample_count][node] = ot_predict[sample_count]

    return it_before


def svm_iteration(it_before, ot_before_ground, num_nodes, graph_init, num_train):
    """
    Parameters
    ----------
    it_before: list
    ot_before_ground: list
    graph_init: graph
    num_nodes: int
    num_train: int
    Returns
    ---------
    it_before: list
    """
    lr_it = samples_each_node_for_iteration(it_before, ot_before_ground, num_nodes, graph_init, num_train)[0]

    for node in range(num_nodes):

        in_neighbors = list(graph_init.predecessors(node))
        if len(in_neighbors) == 0:
            continue

        arr_it = np.array(lr_it[node])
        # print(arr_it)
        # print(arr_it)
        model = pickle.load(open(f'svm_models/{node}_lr_model.sav', 'rb'))

        ot_predict = model.predict(arr_it)

        for sample_count in range(num_train):
            it_before[sample_count][node] = ot_predict[sample_count]

    return it_before


def lr_accuracy(it_before, ot_before_ground, num_nodes, num_train):
    """
    Parameters
    ----------
    it_before: list
    ot_before_ground: list
    num_nodes: int
    num_train: int
    Returns
    ---------
    accuracy: float
    """

    error = 0
    for each_sample in range(num_train):
        for each_node in range(num_nodes):
            if it_before[each_sample][each_node] != ot_before_ground[each_sample][each_node]:
                error += 1
    accuracy = error/(num_train * num_nodes)

    return accuracy


def lr_experiments(filename_sample, g_init, num_sample, num_train, num_test, t_iteration, num_cascades, num_nodes):
    """
    Parameters
    ----------
    filename_sample: str
    g_init: graph
    num_sample: int
    num_train: int
    num_test: int
    t_iteration: int
    num_cascades: int
    num_nodes: int
    Returns
    ---------
    train_accuracy: float
    test_accuracy: float
    """

    initial_train_status, train_t_list = lr_model(filename_sample, num_sample, g_init, num_train, num_cascades, num_nodes)
    a, initial_test_status, c, d, test_t_list, f = choose_samples(filename_sample, num_sample, g_init, num_test, num_cascades, num_nodes)

    lr_train_it, train_ground = lr_first_iteration(initial_train_status, num_nodes, g_init, num_train, train_t_list, num_cascades)
    lr_test_it, test_ground = lr_first_iteration(initial_test_status, num_nodes, g_init, num_test, test_t_list, num_cascades)

    train_accuracy = 0
    test_accuracy = 0

    for i in range(t_iteration):
        if i == 0:
            train_accuracy = lr_accuracy(lr_train_it, train_ground, num_nodes, num_train)
            test_accuracy = lr_accuracy(lr_test_it, test_ground, num_nodes, num_test)

        else:
            lr_train_ot = lr_iteration(lr_train_it, train_ground, num_nodes, g_init, num_train)
            train_accuracy = lr_accuracy(lr_train_ot, train_ground, num_nodes, num_train)

            lr_test_ot = lr_iteration(lr_test_it, test_ground, num_nodes, g_init, num_test)
            test_accuracy = lr_accuracy(lr_test_ot, test_ground, num_nodes, num_test)

    return train_accuracy, test_accuracy


def svm_experiments(filename_sample, g_init, num_sample, num_train, num_test, t_iteration, num_cascades, num_nodes):
    """
    Parameters
    ----------
    filename_sample: str
    g_init: graph
    num_sample: int
    num_train: int
    num_test: int
    t_iteration: int
    num_cascades: int
    num_nodes: int
    Returns
    ---------
    train_accuracy: float
    test_accuracy: float
    """

    initial_train_status, train_t_list = svm_model(filename_sample, num_sample, g_init, num_train, num_cascades, num_nodes)
    a, initial_test_status, c, d, test_t_list, f = choose_samples(filename_sample, num_sample, g_init, num_test, num_cascades, num_nodes)

    lr_train_it, train_ground = svm_first_iteration(initial_train_status, num_nodes, g_init, num_train, train_t_list, num_cascades)
    lr_test_it, test_ground = svm_first_iteration(initial_test_status, num_nodes, g_init, num_test, test_t_list, num_cascades)

    train_accuracy = 0
    test_accuracy = 0

    for i in range(t_iteration):
        if i == 0:
            train_accuracy = lr_accuracy(lr_train_it, train_ground, num_nodes, num_train)
            test_accuracy = lr_accuracy(lr_test_it, test_ground, num_nodes, num_test)

        else:
            lr_train_ot = svm_iteration(lr_train_it, train_ground, num_nodes, g_init, num_train)
            train_accuracy = lr_accuracy(lr_train_ot, train_ground, num_nodes, num_train)

            lr_test_ot = svm_iteration(lr_test_it, test_ground, num_nodes, g_init, num_test)
            test_accuracy = lr_accuracy(lr_test_ot, test_ground, num_nodes, num_test)

    return train_accuracy, test_accuracy


def random_guess(sample_filename, number_samples, guess_sample_num, graph, t_diffusion, t_choose, num_cascades, num_nodes):
    """
    Parameters
    ----------
    sample_filename: str
    number_samples: int
    guess_sample_num: int
    graph: graph
    t_diffusion: int
    t_choose: int
    num_cascades: int
    num_nodes: int
    Returns
    ---------
    accuracy: float
    """
    a, b, c, d, h = choose_samples(sample_filename, number_samples, graph, guess_sample_num, t_diffusion, t_choose, num_cascades, num_nodes)
    it_before, ot_before_ground = samples_for_iteration(b, num_nodes, guess_sample_num, t_choose, num_cascades)


    error = 0

    for each_sample in range(guess_sample_num):
        for each_node in range(num_nodes):
            it_before[each_sample][each_node] = random.randint(0, 3)
            if it_before[each_sample][each_node] != ot_before_ground[each_sample][each_node]:
                error += 1
    accuracy = error / (guess_sample_num * num_nodes)

    return accuracy



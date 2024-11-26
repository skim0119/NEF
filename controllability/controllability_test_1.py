import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# WRG gaussian weight added
def weighted_random(n, p, mean, std):
    graph = nx.erdos_renyi_graph(n, p)

    weight_list = []
    for u, v in graph.edges():
        noise = np.random.uniform(0, 1e-7)
        graph[u][v]["weight"] = np.random.geometric(p) - 1 + noise
        weight_list.append((graph[u][v]["weight"], u, v))

    weight_list.sort(key=lambda x: x[0])
    gaussian_weights = np.random.normal(mean, std, len(graph.edges()))
    gaussian_weights.sort()

    for (old_weight, u, v), new_weight in zip(weight_list, gaussian_weights):
        graph[u][v]["weight"] = new_weight

    return graph


def ring_lattice(n, k, mean, std):
    graph = nx.watts_strogatz_graph(n, k, 0)

    weight_list = []
    for u, v in graph.edges():
        noise = np.random.uniform(0, 1e-7)
        graph[u][v]["weight"] = 1 / abs(u - v) + noise
        weight_list.append((graph[u][v]["weight"], u, v))

    weight_list.sort(key=lambda x: x[0])
    gaussian_weights = np.random.normal(mean, std, len(graph.edges()))
    gaussian_weights.sort()

    for (old_weight, u, v), new_weight in zip(weight_list, gaussian_weights):
        graph[u][v]["weight"] = new_weight

    return graph


def watts_strogatz(n, k, p, mean, std):
    graph = nx.watts_strogatz_graph(n, k, p)

    weight_list = []
    for u, v in graph.edges():
        noise = np.random.uniform(0, 1e-7)
        graph[u][v]["weight"] = 1 / abs(u - v) + noise
        weight_list.append((graph[u][v]["weight"], u, v))

    weight_list.sort(key=lambda x: x[0])
    gaussian_weights = np.random.normal(mean, std, len(graph.edges()))
    gaussian_weights.sort()

    for (old_weight, u, v), new_weight in zip(weight_list, gaussian_weights):
        graph[u][v]["weight"] = new_weight

    return graph


def random_geometric(n, p, mean, std):
    graph = nx.random_geometric_graph(n, 1)
    k = int(n * p)
    weight_list = []
    for u, v in graph.edges():
        noise = np.random.uniform(0, 1e-7)
        pos = nx.get_node_attributes(graph, "pos")
        dist = ((pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2) ** 0.5
        graph[u][v]["weight"] = 1 / dist + noise
        weight_list.append((graph[u][v]["weight"], dist, u, v))

    weight_list.sort(key=lambda x: x[1])

    for weight, dist, u, v in weight_list[k:]:
        graph.remove_edge(u, v)

    gaussian_weights = np.random.normal(mean, std, len(graph.edges()))
    gaussian_weights.sort()

    for (old_weight, old_dist, u, v), new_weight in zip(weight_list[:k], gaussian_weights):
        graph[u][v]["weight"] = new_weight

    return graph


def barabasi_albert(n, k, mean, std):
    graph = nx.barabasi_albert_graph(n, k)

    weight_list = []
    for u, v in graph.edges():
        noise = np.random.uniform(0, 1e-7)
        graph[u][v]["weight"] = (graph.degree[u] + graph.degree[v]) / 2 + noise
        weight_list.append((graph[u][v]["weight"], u, v))

    weight_list.sort(key=lambda x: x[0])
    gaussian_weights = np.random.normal(mean, std, len(graph.edges()))
    gaussian_weights.sort()

    for (old_weight, u, v), new_weight in zip(weight_list, gaussian_weights):
        graph[u][v]["weight"] = new_weight

    return graph


def compute_controllability(graph_type, num_nodes, num_iter):
    global_controllability_list2 = []
    ave_controllability_list2 = []
    modal_controllability = np.zeros((num_iter, num_nodes))

    for i in range(num_iter):
        if graph_type == "WRG":
            graph = weighted_random(num_nodes, 0.2, mean=0.5, std=0.12)
        elif graph_type == "RL":
            graph = ring_lattice(num_nodes, 30, mean=0.5, std=0.12)
        elif graph_type == "WS":
            graph = watts_strogatz(num_nodes, 30, 0.4, mean=0.5, std=0.12)
        elif graph_type == "RG":
            graph = random_geometric(num_nodes, 15, mean=0.5, std=0.12)
        elif graph_type == "BA":
            graph = barabasi_albert(num_nodes, 15, mean=0.5, std=0.12)

        # Matrix A
        mat_adjacent = nx.adjacency_matrix(graph).todense()
        eigenvalues_adjacent, eigenvectors_adjacent = np.linalg.eig(mat_adjacent)
        mat_adjacent_re = mat_adjacent / (1 + np.max(eigenvalues_adjacent))
        eigenvalues_adjacent_re, eigenvectors_adjacent_re = np.linalg.eig(mat_adjacent_re)
        eigenvalues_adjacent_squared = np.abs(eigenvalues_adjacent_re) ** 2

        global_controllability_list = []
        ave_controllability_list = []
        for node in range(num_nodes):
            # Matrix B
            mat_input = np.zeros((num_nodes, 1))
            mat_input[node][0] = 1

            # Check maximum stable step.
            max_step = 0
            for step in range(1, num_nodes + 1):
                controllability_matrix = np.hstack(
                    [
                        np.linalg.matrix_power(mat_adjacent_re, i) @ mat_input
                        for i in range(step)
                    ]
                )
                rank = np.linalg.matrix_rank(controllability_matrix)
                if rank == step:
                    continue
                else:
                    max_step = step
                    break

            # Compute controllability grammian
            grammian = np.zeros((num_nodes, num_nodes))
            adjacency_t = np.eye(num_nodes)
            for t in range(max_step):
                adjacency_t = np.linalg.matrix_power(mat_adjacent_re, t)
                grammian += adjacency_t @ mat_input @ mat_input.T @ adjacency_t.T

            # Ensure it is symmetric
            grammian = (grammian + grammian.T) / 2
            eigenvals_wc, _ = np.linalg.eigh(grammian)
            global_controllability_list.append(np.min(abs(eigenvals_wc)))

            mat_unit = np.eye(mat_adjacent_re.shape[0])
            mat_b = mat_input @ mat_input.T
            mat_a = np.linalg.inv(mat_unit - np.linalg.matrix_power(mat_adjacent_re, 2))
            ave_controllability_list.append(np.trace(mat_a @ mat_b))

            phi = 0
            for mode in range(num_nodes):
                phi += (1 - eigenvalues_adjacent_squared[mode]) * (
                        eigenvectors_adjacent_re[node, mode] ** 2
                )
            modal_controllability[i, node] = phi

        global_controllability_list2.append(np.mean(global_controllability_list))
        ave_controllability_list2.append(np.mean(ave_controllability_list))

    modal_controllability_ave = np.mean(modal_controllability, axis=1)

    # Compute global controllability
    print(np.max(global_controllability_list2))
    print(np.min(global_controllability_list2))

    # Compute average controllability
    print(np.max(ave_controllability_list2))
    print(np.min(ave_controllability_list2))

    # Compute modal controllability
    print(np.max(modal_controllability_ave))
    print(np.min(modal_controllability_ave))

    return global_controllability_list2, ave_controllability_list2, modal_controllability_ave


num_nodes = 128
num_iter = 50

WRG_glo, WRG_ave, WRG_mod = compute_controllability(graph_type="WRG", num_nodes=num_nodes, num_iter=num_iter)
RL_glo, RL_ave, RL_mod = compute_controllability(graph_type="WRG", num_nodes=num_nodes, num_iter=num_iter)
WS_glo, WS_ave, WS_mod = compute_controllability(graph_type="WRG", num_nodes=num_nodes, num_iter=num_iter)
RG_glo, RG_ave, RG_mod = compute_controllability(graph_type="WRG", num_nodes=num_nodes, num_iter=num_iter)
BA_glo, BA_ave, BA_mod = compute_controllability(graph_type="WRG", num_nodes=num_nodes, num_iter=num_iter)

data1 = [WRG_glo, RL_glo, WS_glo, RG_glo, BA_glo]
data2 = [WRG_ave, RL_ave, WS_ave, RG_ave, BA_ave]
data3 = [WRG_mod, RL_mod, WS_mod, RG_mod, BA_mod]
labels = ['WRG', 'RL', 'WS', 'RG', 'BA']

plt.figure(1, figsize=(10, 6))
plt.boxplot(data1, patch_artist=True, showmeans=True)
plt.xticks(range(1, len(labels) + 1), labels)
plt.ylabel(r'Global Controllability ($\times 10^{-21}$)')
plt.title('Gaussian, 128 nodes')
plt.savefig('Global Controllability.png', dpi=300)

plt.figure(2, figsize=(10, 6))
plt.boxplot(data2, patch_artist=True, showmeans=True)
plt.xticks(range(1, len(labels) + 1), labels)
plt.ylabel(r'Average Controllability ($\times 10^{-21}$)')
plt.title('Gaussian, 128 nodes')
plt.savefig('Average Controllability.png', dpi=300)

plt.figure(3, figsize=(10, 6))
plt.boxplot(data3, patch_artist=True, showmeans=True)
plt.xticks(range(1, len(labels) + 1), labels)
plt.ylabel(r'Modal Controllability ($\times 10^{-21}$)')
plt.title('Gaussian, 128 nodes')
plt.savefig('Modal Controllability.png', dpi=300)
plt.show()

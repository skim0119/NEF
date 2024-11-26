import networkx as nx
import numpy as np


# WRG gaussian weight added
def weighted_random(n, p, reweight, mean, std):
    graph = nx.erdos_renyi_graph(n, p)

    if reweight:
        weight_list = []
        for u, v in graph.edges():
            graph[u][v]["weight"] = np.random.geometric(p)
            weight_list.append((graph[u][v]["weight"], u, v))

        weight_list.sort(key=lambda x: x[0])
        gaussian_weights = np.random.normal(mean, std, len(graph.edges()))
        gaussian_weights.sort()

        for (old_weight, u, v), new_weight in zip(weight_list, gaussian_weights):
            graph[u][v]["weight"] = new_weight

    else:
        for u, v in graph.edges():
            graph[u][v]["weight"] = np.random.geometric(p)
    return graph


def ring_lattice(n, k):
    graph = nx.watts_strogatz_graph(n, k, 0)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1 / abs(u - v)
    return graph


def watts_strogatz(n, k, p):
    graph = nx.watts_strogatz_graph(n, k, p)
    return graph


def random_geometric(n, p):
    graph = nx.random_geometric_graph(n, 1)
    k = int(n * p)
    dist_list = []
    for u, v in graph.edges():
        pos = nx.get_node_attributes(graph, "pos")
        dist = ((pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2) ** 0.5
        dist_list.append((dist, u, v))
        graph[u][v]["weight"] = 1 / dist

    dist_list.sort(key=lambda x: x[0])

    for dist, u, v in dist_list[k:]:
        graph.remove_edge(u, v)

    return graph


def barabasi_albert(n, m):
    graph = nx.barabasi_albert_graph(n, m)
    for u, v in graph.edges():
        graph[u][v]["weight"] = (graph.degree[u] + graph.degree[v]) / 2

    return graph


num_nodes = 128
num_control_nodes = 1
graph = weighted_random(num_nodes, 0.1, reweight=True, mean=0.5, std=0.12)

# Matrix A
adjacent_matrix = nx.adjacency_matrix(graph).todense()
eigenvalues_adjacent, eigenvectors_adjacent = np.linalg.eig(adjacent_matrix)
adjacent_matrix_re = adjacent_matrix / (1 + np.max(eigenvalues_adjacent))
eigenvalues_adjacent_re, eigenvectors_adjacent_re = np.linalg.eig(adjacent_matrix_re)

# Matrix B
input_matrix = np.zeros((num_nodes, num_control_nodes))
for i in range(np.min([num_nodes, num_control_nodes])):
    input_matrix[i][i] = 1

# Check maximum stable step. Ref: <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9414/MediaObjects/41467_2015_BFncomms9414_MOESM132_ESM.pdf> Page 3
max_step = 0
for step in range(1, num_nodes + 1):
    controllability_matrix = np.hstack(
        [
            np.linalg.matrix_power(adjacent_matrix_re, i) @ input_matrix
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
    adjacency_t = np.linalg.matrix_power(adjacent_matrix_re, t)
    grammian += adjacency_t @ input_matrix @ input_matrix.T @ adjacency_t.T

# Ensure it is symmetric
grammian = (grammian + grammian.T) / 2

# Compute global controllability
eigenvals_wc, _ = np.linalg.eigh(grammian)
global_controllability = np.min(abs(eigenvals_wc))
print(f"global controllability:{global_controllability:.1e}")

# Compute average controllability
grammian_inverse = np.linalg.inv(grammian)
ave_controllability = np.trace(grammian)
print(f"average controllability:{ave_controllability}")

# Compute modal controllability
eigenvalues_adjacent_squared = np.abs(eigenvalues_adjacent_re) ** 2
modal_controllability = np.zeros(num_nodes)

for node in range(num_nodes):
    phi = 0
    for mode in range(num_nodes):
        phi += (1 - eigenvalues_adjacent_squared[mode]) * (
            eigenvectors_adjacent_re[node, mode] ** 2
        )
    modal_controllability[node] = phi

modal_controllability_ave = sum(modal_controllability) / num_nodes
print(f"modal controllability:{modal_controllability_ave}")

# print(graph.number_of_edges())
# print(graph.number_of_nodes())
# print(G_WRG.nodes())
# print(G_WRG.edges(data=True))

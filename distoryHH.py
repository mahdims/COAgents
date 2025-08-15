import typing
from typing import List, Tuple
import os
from os import path
import numpy as np
import torch as th

import networkx as nx
def adj_matrix_to_routes(adj_matrix):
    # 1. Build DiGraph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    # 2. Compute in‐ and out‐degree of each node
    indegree  = dict(G.in_degree())
    outdegree = dict(G.out_degree())

    routes = []
    # 3. Build all maximal paths from “starts”
    for u, outd in outdegree.items():
        if outd > 0 and indegree.get(u, 0) != 1:
            for _, v in G.out_edges(u):
                path = [u, v]
                while indegree[v] == 1 and outdegree[v] == 1:
                    w = next(G.successors(v))
                    path.append(w)
                    v = w
                routes.append(path)

    # 4. Add isolated nodes as singleton routes
    for u in G.nodes():
        if indegree.get(u, 0) == 0 and outdegree.get(u, 0) == 0:
            routes.append([u])

    return routes
def route_to_adj_matrix(routes, nNodes):
    target_matrix = np.zeros((nNodes, nNodes))
    for route in routes:
        if len(route)==2: continue
        for i in range(len(route) - 1):
            source = route[i]
            destination = route[i+1]
            target_matrix[source, destination] = 1.0
    return target_matrix
def destroy_edges(curr: np.ndarray,
                  props: np.ndarray,
                  frac: float = None,
                  num_remove: int = None,
                  seed: int = None) -> np.ndarray:

    # find coordinates of existing edges
    coords = np.argwhere(curr == 1)
    total = coords.shape[0]
    if total == 0:
        return curr

    # decide how many to drop
    if frac is not None:
        k = int(frac * total)
    else:
        k = int(num_remove)
    k = max(0, min(k, total))
    if k == 0:
        return curr

    # extract their probabilities
    edge_probs = props[coords[:, 0], coords[:, 1]]

    #  shuffle for tie-breaking
    if seed is not None:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(total)
        coords = coords[perm]
        edge_probs = edge_probs[perm]

    # sort ascending by probability and pick the first k to remove
    order = np.argsort(edge_probs)
    to_remove = coords[order[:k]]

    # remove them
    for i, j in to_remove:
        curr[i, j] = 0

    return curr

def simulate_segment(nodes, D, time_windows, service_times, demands):
    """
    Simulate along a route segment to compute:
      - total load
      - end departure time
      - arrival-time offsets (delta) relative to first node
    Returns a dict with keys: nodes, load, T_end, delta, start, end
    """
    arrival = {}
    t = 0.0
    a = time_windows[:, 0]
    b = time_windows[:, 1]
    for idx, u in enumerate(nodes):
        if idx == 0:
            # if first node, start at its earliest time
            t = max(0.0, a[u])
        else:
            prev = nodes[idx - 1]
            t += D[prev, u]
            t = max(t, a[u])
        arrival[u] = t
        t += service_times[u]

    load = sum(demands[n] for n in nodes)
    first_arr = arrival[nodes[0]]
    delta = [arrival[u] - first_arr for u in nodes]
    return {
        'nodes': list(nodes),
        'load': load,
        'T_end': t,
        'delta': delta,
        'start': nodes[0],
        'end': nodes[-1],
        'arrival': arrival,
    }

def get_candidates(segments, p, D, demands, time_windows, service_times, capacity):
    """
    Identify all feasible edge insertions between segment ends and starts
    (including return-to-depot), returning a sorted list of (probability, si, tj).
    Feasibility is checked by fully simulating each merged route.
    """
    a = time_windows[:, 0]
    b = time_windows[:, 1]

    # Pre-simulate each partial segment
    seg_info = [
        simulate_segment(np.array(seg), D, time_windows, service_times, demands)
        for seg in segments
    ]

    candidates = []
    for si, s in enumerate(seg_info):
        # skip segments that already end at depot
        if s['end'] == 0:
            continue

        for tj, t in enumerate(seg_info):
            # skip segments that start at depot or self-merges
            if t['start'] == 0 or si == tj:
                continue

            i = s['end']
            j = t['start']

            # capacity check on the two pieces
            if s['load'] + t['load'] > capacity:
                continue

            # build the merged node sequence
            merged = s['nodes'] + t['nodes']
            # ensure depot at front and back
            route = merged.copy()
            if route[0] != 0:
                route.insert(0, 0)
            if route[-1] != 0:
                route.append(0)

            # fully simulate the merged route
            info = simulate_segment(np.array(route), D, time_windows, service_times, demands)

            # capacity check on full route
            if info['load'] > capacity:
                continue

            # time-window check on full route (including depot returns)
            bad = False
            for u, arr in info['arrival'].items():
                if arr < a[u] or arr > b[u]:
                    bad = True
                    break
            if bad:
                continue

            # if it survives all checks, it's a valid candidate
            candidates.append((p[i, j], si, tj))

    # sort by descending probability
    candidates.sort(key=lambda x: -x[0])
    return candidates

def complete_solution(segments, p, D, demands, time_windows, service_times, capacity):
    # make a working copy
    routes = [list(seg) for seg in segments]

    # iteratively merge interior segments
    while True:
        candidates = get_candidates(routes, p, D, demands, time_windows, service_times, capacity)
        if not candidates:
            break
        # pick best merge
        _, si, tj = candidates[0]

        # merge segments: s -> t
        seg_s = routes[si]
        seg_t = routes[tj]
        new_seg = seg_s + seg_t

        # remove old segments (pop larger index first)
        for idx in sorted([si, tj], reverse=True):
            routes.pop(idx)
        routes.append(new_seg)

    # ensure all routes start and end at depot
    final_routes = []
    for seg in routes:
        r = seg.copy()
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
        final_routes.append(r)

    return final_routes




def is_feasible_route(route, D, demands, time_windows, service_times, capacity):
    """
    Check capacity, all time‐windows (including return to depot) for a full route.
    """
    info = simulate_segment(np.array(route), D, time_windows, service_times, demands)
    # capacity
    if info['load'] > capacity:
        return False
    # time‐windows
    a, b = time_windows[:,0], time_windows[:,1]
    for u, arr in info['arrival'].items():
        if arr < a[u] or arr > b[u]:
            return False
    # return‐to‐depot
    back = info['T_end'] + D[info['end'], 0]
    if back > b[0]:
        return False
    return True

def complete_solution_global(segments, p, D, demands, time_windows, service_times, capacity):
    """
    Merge partial customer‐only segments into full routes without interior depot visits.
    Returns a list of routes, each starting and ending at depot 0.
    """
    # 1) Strip any depot markers from the input segments
    segs = {i: [u for u in seg if u != 0] for i, seg in enumerate(segments)}

    # 2) Collect all (si, tj) merges with their probabilities
    candidates = []
    for si, seg_s in segs.items():
        for tj, seg_t in segs.items():
            if si == tj:
                continue
            if seg_s[-1] == 0 or seg_t[0] == 0:
                continue  # should never happen since we stripped 0s, but safe
            i, j = seg_s[-1], seg_t[0]
            if sum(demands[n] for n in seg_s) + sum(demands[n] for n in seg_t) > capacity:
                continue
            candidates.append((p[i, j], si, tj))
    candidates.sort(key=lambda x: -x[0])

    # 3) Union‐find to avoid cycles
    parent = {i: i for i in segs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 4) Try each merge once, in descending p
    for prob, si, tj in candidates:
        ri, rj = find(si), find(tj)
        if ri == rj:
            continue
        merged = segs[ri] + segs[rj]
        # full‐route check: add depot at ends for simulation
        route = [0] + merged + [0]
        info = simulate_segment(np.array(route), D, time_windows, service_times, demands)
        if info['load'] > capacity:
            continue
        a, b = time_windows[:,0], time_windows[:,1]
        if any(arr < a[u] or arr > b[u] for u, arr in info['arrival'].items()):
            continue
        # commit the merge
        parent[rj] = ri
        segs[ri] = merged
        del segs[rj]

    # 5) Build final routes
    final_routes = []
    seen = set()
    for i in segs:
        root = find(i)
        if root in seen:
            continue
        seen.add(root)
        route = [0] + segs[root] + [0]
        final_routes.append(route)

    return final_routes


def compute_softdist(D: np.ndarray, tau: float) -> np.ndarray:

    # avoid dividing by zero on the diagonal
    D_safe = D.copy()
    np.fill_diagonal(D_safe, np.inf)
    
    # exponentiate with temperature
    exp_neg = np.exp(-D_safe / tau)
    
    # normalize rows (excluding diagonal)
    row_sums = exp_neg.sum(axis=1, keepdims=True)
    Φ = exp_neg / row_sums
    
    # ensure we zero out self‐scores
    np.fill_diagonal(Φ, 0.0)
    return Φ


def compute_global_softdist(D: np.ndarray, tau: float) -> np.ndarray:

    # Copy and zero out diagonal so self-edges have zero weight
    D_safe = D.copy()
    np.fill_diagonal(D_safe, np.inf)

    # Compute exponentiated scores
    exp_neg = np.exp(-D_safe / tau)

    # Normalize across all entries
    total = exp_neg.sum()
    Ψ = exp_neg / total

    # Zero out diagonal explicitly
    np.fill_diagonal(Ψ, 0.0)
    return Ψ




def build_combined_score(p: np.ndarray,
                         D: np.ndarray,
                         tau: float,
                         alpha: float = 0.5) -> np.ndarray:

    # compute SoftDist heatmap
    Φ = compute_softdist(D, tau) #compute_global_softdist(D, tau)#
    Φ[p == 0] = 0 
    # convex combination
    score = alpha * p + (1 - alpha) * Φ
    return score

def edge_unlikely_string_removal(
    routes: List[List[int]],
    heatmap: np.ndarray,
    max_length: int,
    num_strings_to_remove: int
) -> Tuple[
    List[List[int]],                 # new_routes
    List[List[int]],                 # removed_strings_customers
    List[Tuple[float, int, int, int, List[int]]]  # removed_strings_info
]:

    rng = np.random.default_rng()
    inv = 1.0 - heatmap  # precompute inverse probabilities

    # 1. Collect all candidate windows
    candidates: List[Tuple[float, int, int, int]] = []  # (score, route_idx, start_idx, length)
    for r_idx, route in enumerate(routes):
        m = len(route)
        # Must have at least depot + one customer + depot
        if m < 3:
            continue
        interior_count = m - 2
        # sample length from 1 to min(max_length, interior_count)
        L = rng.integers(1, min(max_length, interior_count) + 1)
        
        if L == 1:
            # singleton removal score
            for pos in range(1, m - 1):
                u, v, w = route[pos - 1], route[pos], route[pos + 1]
                score = (inv[u, v] + inv[v, w]) / 2.0
                candidates.append((score, r_idx, pos, 1))
        else:
            # vectorized sliding-window for L > 1
            route_arr = np.array(route)
            edges_inv = inv[route_arr[:-1], route_arr[1:]]  # length m-1
            cumsum = np.concatenate(([0.0], edges_inv.cumsum()))  # length m
            # interior windows only: start from 1 to m-L-1
            for start in range(1, m - L):
                sum_inv = cumsum[start + L - 1] - cumsum[start]
                avg_score = sum_inv / (L - 1)
                candidates.append((avg_score, r_idx, start, L))

    # 2. Sort by descending score
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 3. Select top-k non-overlapping strings
    removed_customers = set()
    removed_strings_customers: List[List[int]] = []
    removed_strings_info: List[Tuple[float, int, int, int, List[int]]] = []
    selected = 0

    for score, r_idx, start, L in candidates:
        if selected >= num_strings_to_remove:
            break
        nodes = routes[r_idx][start:start + L]
        # Ensure no overlap: skip if any node already targeted
        if any(c in removed_customers for c in nodes):
            continue
        # Register selection
        removed_strings_customers.append(nodes)
        removed_strings_info.append((score, r_idx, start, start + L, nodes))
        removed_customers.update(nodes)
        selected += 1

    # 4. Rebuild routes, dropping empty or [0,0]
    new_routes: List[List[int]] = []
    for route in routes:
        filtered = [c for c in route if c not in removed_customers]
        if not filtered or filtered == [0, 0]:
            continue
        new_routes.append(filtered)

    return new_routes, removed_strings_customers, removed_strings_info

def compute_ruin_parameters(
    routes: List[List[int]],
    L_max: int,
    c_bar: float,
    rng: np.random.Generator = None
) -> Tuple[int, float, List[int], List[int]]:

    if rng is None:
        rng = np.random.default_rng()

    # Compute interior customer counts per route (exclude depots at start/end if present)
    interior_counts = []
    for route in routes:
        if len(route) >= 2 and route[0] == 0 and route[-1] == 0:
            interior_counts.append(len(route) - 2)
        else:
            interior_counts.append(len(route))

    # Consider only tours with at least one customer
    valid_indices = [i for i, cnt in enumerate(interior_counts) if cnt > 0]
    T = len(valid_indices)
    if T == 0:
        return 0, 0.0, [], []

    # 1. Global max string size ell_max_s
    avg_size = sum(interior_counts[i] for i in valid_indices) / T
    ell_max_s = min(L_max, avg_size)

    # 2. Maximum number of strings k_max_s
    k_max_s = int(np.floor(2 * c_bar / (1 + ell_max_s)))
    k_max_s = max(1, k_max_s)

    # 3. Sample actual number of strings k_s
    k_s = int(rng.integers(1, k_max_s + 1))

   

    return k_s, ell_max_s

def route_load(route: List[int], demands: np.ndarray):
    """Compute total demand of a route """
    return sum(demands[c] for c in route)

def is_time_feasible(route: List[int],
                     time_windows: np.ndarray,
                     service_times: np.ndarray,
                     travel_times: np.ndarray) -> bool:
    """
    Check time-window feasibility of a complete route (including depot at start/end).
    """
    current_time = 0.0
    prev = route[0]
    # Wait until depot earliest if needed
    if current_time < time_windows[prev, 0]:
        current_time = time_windows[prev, 0]
    if current_time > time_windows[prev, 1]:
        return False
    # Traverse through route
    for node in route[1:]:
        current_time += service_times[prev] + travel_times[prev, node]
        # if early, wait
        if current_time < time_windows[node, 0]:
            current_time = time_windows[node, 0]
        if current_time > time_windows[node, 1]:
            return False
        prev = node
    return True

def heatmap_regret_repair(
    routes: List[List[int]],
    removed: List[int],
    heatmap: np.ndarray,
    demands: np.ndarray,
    time_windows: np.ndarray,
    service_times: np.ndarray,
    travel_times: np.ndarray,
    vehicle_capacity: int
) -> List[List[int]]:
    removed = removed.copy()
    while removed:
        insertion_options = {}
        # Evaluate insertions into both existing routes and a new route
        for c in removed:
            best_score = -np.inf
            second_score = -np.inf
            best_route_idx, best_pos = None, None

            # Existing routes
            for r_idx, route in enumerate(routes):
                # capacity constraint
                if route_load(route, demands) + demands[c] > vehicle_capacity:
                    continue
                for pos in range(1, len(route)):
                    i, j = route[pos-1], route[pos]
                    score = heatmap[i, c] + heatmap[c, j]
                    new_route = route[:pos] + [c] + route[pos:]
                    if not is_time_feasible(new_route, time_windows, service_times, travel_times):
                        continue
                    # update top two scores
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_route_idx, best_pos = r_idx, pos
                    elif score > second_score:
                        second_score = score

            # New route option
            # treat new route as index len(routes), position 1
            if demands[c] <= vehicle_capacity:
                score_new = heatmap[0, c] + heatmap[c, 0]
                if is_time_feasible([0, c, 0], time_windows, service_times, travel_times):
                    # update top two scores with new-route score
                    if score_new > best_score:
                        second_score = best_score
                        best_score = score_new
                        best_route_idx, best_pos = len(routes), 1
                    elif score_new > second_score:
                        second_score = score_new

            # record option if any valid insertion found
            if best_route_idx is not None:
                regret = best_score - (second_score if second_score > -np.inf else 0.0)
                insertion_options[c] = (regret, best_score, best_route_idx, best_pos)

        # pick customer with max regret
        c_star, (regret, score, r_idx, pos) = max(insertion_options.items(), key=lambda x: x[1][0])
        #print(f"Inserting customer {c_star} with regret {regret:.4f}, score {score:.4f} into route {r_idx} at pos {pos}")

        # perform insertion
        if r_idx == len(routes):
            # new route
            routes.append([0, c_star, 0])
        else:
            routes[r_idx].insert(pos, c_star)

        removed.remove(c_star)

    return routes
def ruin_repair(
    routes: List[List[int]],
    heatmap: np.ndarray,
    demands: np.ndarray,
    time_windows: np.ndarray,
    service_times: np.ndarray,
    travel_times: np.ndarray,
    vehicle_capacity: int,
    L_max: int = 10,
    c_bar: float = 10.0
) -> List[List[int]]:
    """
    Full pipeline: compute ruin parameters, apply ruin, then repair, returning final routes.
    """
    k_s, ell_max_s = compute_ruin_parameters(routes, L_max, c_bar)
    max_length = int(np.floor(ell_max_s))
    # Ruin
    new_routes, removed_strings, _ = edge_unlikely_string_removal(routes, heatmap, max_length, k_s)
    removed_customers = [c for s in removed_strings for c in s]
    # Repair
    final_routes = heatmap_regret_repair(
        new_routes, removed_customers, heatmap,
        demands, time_windows, service_times, travel_times,
        vehicle_capacity
    )
    return final_routes
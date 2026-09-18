# COAgents: Method Notes

Working notes on how the COAgents method actually works, derived from reading the code
(not from a paper). File and line references point at the implementation so claims can be
re-checked. Where the code and its comments disagree, the code wins and the discrepancy is
flagged in section 12.

---

## 1. One-paragraph summary

COAgents is a **learned hyper-heuristic** for VRPTW / CVRP. A classical local-search
solver is driven by neural "agents" that read the **history of the search itself**, encoded
as a graph of visited solutions (a *partial search graph*, PSG), and decide:

1. **which previously visited solution to branch from** (Node Selection),
2. **which of 19 low-level improvement heuristics to apply to it** (Move Selection), and
3. **when and where to jump** when the current basin is exhausted (Jump), by predicting an
   edge heat-map of a good solution and either rebuilding a full solution from it or running
   a constrained beam search over a few routes.

Agents 1 and 2 are a single network (`HeuristicNet`, the "moves model"). Agent 3 is a
second network (`E2ENet`, the "heat-map / jumps model") plus non-learned decoders. Both
networks share the same backbone and consume the same history representation.

---

## 2. Search-state representation

### 2.1 A single solution as a `Sample` (`historyHH.embed_sample`)

Every visited solution is embedded into a `Sample` (`model/datastructures.py:33`) with
three data blocks:

| Field | Shape | Meaning |
|---|---|---|
| `u` | `1 × 16` | **Global** (per-solution) features: 8 raw + 8 positional |
| `v` | `nNodes × 70` | **Local** (per-VRP-node) features: 6 raw + 64 positional |
| `t` | `nNodes-1 + 2·nVehicles` | Flattened route sequence (depot-delimited), padded with depot pairs |
| `e` | `m × 3` | Edges `(parent_idx, child_idx, heuristic_id)` in the search graph |
| `n`, `m`, `l` | ints | #solutions in graph, #edges, #VRP nodes per solution |
| `s` | int | `PARAM_NEDGES_TYPE = 19` heuristics |

Raw `u` features (`historyHH.py:36-43`, updated in `HistoryHH.update`):

| idx | value |
|---|---|
| 0 | total cost of the solution |
| 1 | number of routes |
| 2 | number of customers |
| 3 | capacity / total demand |
| 4 | cost gap to parent (parent cost − own cost) |
| 5 | sum of children's costs |
| 6 | sum of squared (child cost − own cost) |
| 7 | number of children + 1 |
| 8..15 | random-walk landing probabilities on the search graph (GraphGPS-style RWPE, `layerGraphEmbedding.get_rw_landing_probs`) |

Raw `v` features per VRP node (`historyHH.py:48-77`): `x, y, demand/capacity, tw_open,
tw_close, service_time`. For CVRP the time columns are zero. The 64-d positional part is a
**cyclic positional encoding** (DACT-style CPE from `layerCyclicFeatureEmbedding`) indexed
by the node's position in the route sequence, so the *same instance* under two different
solutions produces different `v` blocks. Dummy depot slots are trimmed so `l = nNodes`.

### 2.2 The partial search graph (PSG) and `HistoryHH`

`HistoryHH` (`historyHH.py:166`) is a `list[Sample]`. **Each element is one basin**: the
tree of solutions reached by local moves from one starting point. The list grows by one
element on every jump.

`HistoryHH.update(vertex, heuristic, sample)`:
- `heuristic == 19` (a jump): the *previous* active graph is **purged** to the root-to-best
  path (`purge_sample`), the new solution starts a new element. If `len > MAX_SAMPLES (32)`
  one old element is deleted (see quirk 12.a).
- otherwise (a local move): the new solution is appended as a node of the **last** element,
  with edge `(vertex → new, heuristic)`, and the parent's `u[5..7]` and everybody's RWPE
  are refreshed.

`HistoryHH.sample()` concatenates *all* elements into one `Sample` with block sizes `n`
(one entry per basin) and offset-shifted edges. The networks then treat each basin as one
attention block.

**PSG3 vs PSG2** (`HistoryHH.psg3to2`): PSG3 is the full tree of the active basin. PSG2
drops *uphill dead-end leaves*: an edge survives if it is downhill (parent cost > child cost)
or the child itself has children; children of dropped edges are removed. The moves model is
called on PSG2 (`HyperHeuristic.py:562`) and predictions are inflated back to PSG3 indices
via `vertices_map`. Historical basins are already chains after purging, so the difference
only matters for the active basin.

---

## 3. Neural networks

### 3.1 Shared backbone (both models)

```
v_raw(6)  --CyclicEmbedding.embedder--> 32  ⊕ pos(64)  = v (96)
u_raw(8) ⊕ rwpe(8) --GraphEmbeddingLayer MLP--> u (64);  edge type id --Embedding--> ez (32)

repeat 3×:
  (u, v) = GatedGCNLayer(u, v, ez, e)           # message passing over the *search graph*
  v      = MaskedBlockTransformer(u, v)          # attention over solutions *within a basin*
```

- `GatedGCNLayer` (`model/layerGatedGCN.py`): first compresses each solution's `l` node
  vectors together with `u` into one vector `z` (1-D "convolution" = mean of an MLP over
  concatenated `[u, v_i]`), then gates messages along search-graph edges with
  `tanh(K z_i + Q z_j + E e_ij)` in both directions, and de-convolves back to per-node `v`
  and per-solution `u`. So information flows *between solutions* along "which heuristic
  produced which".
- `MaskedBlockTransformer` (`model/layerMaskedBlockTransformer.py`): builds a per-solution
  query/key from `[u, v]`, attends across solutions **only inside the same basin** (block
  mask from `n`), and mixes the per-node value vectors accordingly. 16 heads, z-dim 256.

### 3.2 Moves model: `HeuristicNet` (`model/model.py`)

Head `LossBCELayer` (`model/layerLossBCE.py`): per solution, `z = mean_i MLP([u, v_i])`
(dim 128). Two sigmoid classifiers:
- `ZV(z)` → **P(solution is on the shortest path to the local optimum)** — the *node* score.
- `ZE([z, ee_k])` for each of the 19 heuristic embeddings `ee_k` → **P(heuristic k is the
  best next move from this solution)** — the *edge* score.

At inference (`loss_solutions = loss_heuristics = False`) it returns the flat
concatenation `[n·1 node scores, n·19 edge scores]` (`layerLossBCE.py:153`).

Training labels (only visible in `model.py:118-128`; no training script is in the repo):
`label_solutions = (g_dst == min g_dst in basin)` where `g_dst` is distance to the local
optimum, and `label_heuristics = one_hot(m_bst) * label_solutions`. Loss is class-rebalanced
BCE.

### 3.3 Jump / heat-map model: `E2ENet` (`model/layerE2E.py`)

Same backbone, but in each layer a **DACT encoder** (`layer_DACT.DACTEncoder`, dual-aspect
node/positional attention) also runs on the per-node vectors, and a `DACTDecoder` produces,
for **every solution in the sample**, an `l × l` matrix of logits. After sigmoid this is a
**probabilistic adjacency matrix** ("heat-map") `P[i, j] = P(edge i→j is in a good
solution)`. Trained with `BCEWithLogitsLoss(pos_weight=60)` against a ground-truth adjacency
`t` (`layerE2E.py:147`). Checkpoints: `model/{vrptw,cvrp}/checkpointsE2E/`.

---

## 4. The three agents, mapped to code

| Agent | Code | Input | Output |
|---|---|---|---|
| Node selection | `HyperHeuristic.predictMove` (`HyperHeuristic.py:291`) | `history.sample()` → PSG2 → `HeuristicNet` | index of a solution in the **active** basin |
| Move selection | same call | same forward pass | heuristic id ∈ [0, 18] |
| Jump | `predictJump` (`:370`) / `predictBeam` (`:353`) | `history.sample()` → `E2ENet` | a brand-new `Solution` |

### 4.1 Node + move selection (`predictMove`)

1. Forward PSG2 through the moves model; inflate to PSG3 indices.
2. **Masks**:
   - all nodes from *previous* basins (`node_predictions[:local_area] = 0`) — you can only
     branch inside the active basin;
   - `(node, heuristic)` pairs already expanded from that node (edge already exists);
   - nodes that are **uphill children** (cost ≥ parent's cost): all their moves are zeroed.
3. `conditional = node_p[:, None] * edge_p`; pick `argmax` over the `(node, heuristic)`
   grid. This is a greedy, deterministic decode.
4. **Fallback ladder** if the argmax is 0: uniform node scores → random edge scores with the
   same masks → return `(-1, -1, -1)`, which the main loop interprets as "basin exhausted,
   force a jump".

(Several alternative decoders exist — `decodeBasedOnEdgePrediction`, `simpleDecoding`,
`greedyDecoding` — but none are called; they rely on `self.test_data` which is never set.)

### 4.2 Jump (`predictJump`)

Run `E2ENet` on the whole history → one heat-map per stored solution.

- **`simple`** (first two jumps): take the heat-map of the **best solution in the active
  basin** and decode it greedily with `construct_routes_from_predictions`
  (`model/inferance.py:243`): from the current node, mask infeasible customers (visited,
  capacity, time window, return-to-depot), go to `argmax P[cur, ·]`, close the route when
  nothing is feasible. Feasibility-safe, purely greedy.
- **`hellinger`** (third jump onwards): aim for **diversity**. Build the binary adjacency of
  the best solution in every basin (`reference_solutions`). Weight each reference by the
  inverse of its total cosine similarity to the others (`w_k`) so clusters of near-identical
  minima do not dominate. For every candidate heat-map compute a weighted generalised
  Hellinger distance to the reference set and pick the **most distant** heat-map, then
  decode it greedily as above. Intuition: jump *into* the region of solution space the
  search has not yet visited.

### 4.3 Constrained beam search (`predictBeam` + `beamHH.constrainedBeamHH`)

Used only after iteration 127 and only when a jump would otherwise be triggered.

1. Heat-map `P` of the best solution in the active basin; current best route array `t`.
2. Per-edge **flip probability** `|A − P|` (A = current adjacency). Sum it per pair of routes
   into `pdag`, normalise by route sizes, and sample **3 routes** to mutate with probability
   proportional to their flip mass (`beamHH.py:165-187`).
3. `partial_beam_search_vrp`: keep the other routes fixed; rebuild the 3 routes' customers
   with a stochastic beam of width 256. Extensions are sampled without replacement with
   probability `softmax(P[prev, cand])` over feasible candidates; states whose partial cost
   already exceeds the best complete cost are pruned.
4. Return the best complete reconstruction, or `inf` (→ fall back to `predictJump`).

Note the ruin-and-repair alternative in `distoryHH.py` (`predictdistroy`: drop the
lowest-probability 30–50% of current edges, re-merge segments by heat-map score) is
implemented but **not called** from `solve()`.

---

## 5. The 19 low-level heuristics (`historyHH.applyHeuristics`)

| id | move | implementation |
|---|---|---|
| 0 | shift (relocate a customer) | in-house, `VRPSolution` |
| 1 | interchange (swap two customers) | in-house |
| 2 | 2-opt (intra-route) | in-house |
| 3 | cross-exchange, segment length ≤ 7 | in-house |
| 4 | 2-opt* (inter-route) | in-house |
| 5 | path relocation | in-house |
| 6 | Or-opt | in-house |
| 7–15 | Exchange(1,0) (2,0) (3,0) (1,1) (2,1) (3,1) (2,2) (3,2) (3,3) | PyVRP `LocalSearch` node operators |
| 16 | SwapTails | PyVRP node operator |
| 17 | SwapRoutes | PyVRP route operator (`intensify`) |
| 18 | SwapStar | PyVRP route operator (`intensify`) |

Each heuristic runs to its own local optimum on a **copy** of the selected solution
(`_run_node_op` calls a full `LocalSearch.search`). One "move" in the PSG is therefore one
full local-search pass with one operator, not a single neighbourhood step.

---

## 6. Main algorithm (`HyperHeuristic.solve`, `HyperHeuristic.py:496`)

```
savings-method initial solution
intialPhase:
    3× ALNSLocal (adaptive roulette over the same 19 heuristics), 8 iterations each,
    different seeds; keep the best run and replay its trajectory into history as the
    first PSG (a chain).
best = argmin cost in history[-1]

loop until time ≥ 2000 s  or  iter ≥ 1000  or  gap-to-BKS ≤ 0.1 %:
    iter += 1 ; no_update += 1

    if no_update ≥ 12  or  |active basin| == MAX_SOLUTIONS (96):        # heat-map agents
        if iter > 127 and predictBeam(bestRoute) succeeds:  sol = beam result
        else:                                               sol = predictJump('simple' if no_jumps < 2 else 'hellinger')
        history.update(-1, 19, embed(sol))        # opens a new basin, purges the old one
        no_update = 0 ; no_jumps += 1
    else:                                                                 # hyper-heuristic agent
        (local_area, node, move) = predictMove("PSG2")
        if all == -1: no_update = inf ; continue  # nothing left to try → jump next iter
        sol = history.get_solution(active, node - local_area)   # rebuild from t
        applyHeuristics(move, sol)
        history.update(node - local_area, move, embed(sol))
        if cost < basin's local min: no_update = 0
        if cost < global best: update best
```

Points worth internalising:
- **No acceptance criterion.** `alwaysAccept = True`; every generated solution is stored as
  a graph node. Selection pressure comes entirely from the moves model choosing where to
  branch and from the uphill-child mask. The SA `temperature` is computed but unused.
- **"Stuck" is measured against the basin's own local minimum**, not the global best.
- The global best is tracked as a raw route array (`bestRoute`) and only wrapped back into a
  `Solution` at the very end.
- `MAX_SOLUTIONS = 96` bounds the active basin; `MAX_SAMPLES = 32` bounds the number of
  basins kept, so the network input is bounded in size.
- Both networks are invoked on the **entire** history every iteration (one forward pass
  each); there is no caching of embeddings.

---

## 7. Hyper-parameters (all hard-coded)

| where | name | value |
|---|---|---|
| `RunHH.runHH` | time limit / iteration limit | 2000 s / 1000 |
| `HyperHeuristic.__init__` | `phase1Iter` (ALNSLocal iters) | 8 |
| | `maxPSGsamples` (basins kept) | 32 |
| | `maxPSGsolutions` (nodes per basin) | 96 |
| | beam width | 256 |
| `solve` | stagnation threshold `no_update` | 12 |
| | beam enabled after iteration | 127 |
| | simple → Hellinger jumps after | 2 jumps |
| `constrainedBeamHH` | routes mutated per beam | 3 |
| `predictdistroy` (unused) | destroy fraction | 0.3 / 0.5 |
| `model.py` | u / v / ez / ggcn-z / trans-z | 64 / 96 / 32 / 128 / 256 |
| | heads, layers | 16 heads, 3 GGCN + 3 transformer (+3 DACT in E2E) |
| `datastructures.py` | `PARAM_NEDGES_TYPE` | 19 |
| | `VRP_MAX_NODES`, `VRP_MAX_DUMMY_DEPOTS` | 1250, 128 |
| `stoppingCriteriaMet` | gap tolerance to BKS | 0.001 |

The BKS is read from `dataset/<set>/BestObj.json` and **is part of the stopping rule**, so
reported times depend on knowing the best-known cost.

---

## 8. Runtime and entry points

- `pRunHH.py` — master/worker over shared memory. One worker per GPU; each loads both
  checkpoints once, then pulls instance ids from a `(mgpus+1) × 5` float64 shared array
  guarded by per-worker locks, calls `RunHH.runHH`, writes cost/gap/time back. Master appends
  rows to the CSV log (`name, bestNVehicles, bestCost, algBestCost, algNVehicles, gap, time`).
- `RunHH.runHH(fileName, model, modelE2E, data_dir, BKS, problem_type, device_id)` —
  reads a Solomon-format instance (`InstanceReader`, builds PyVRP data), runs one
  `HyperHeuristic`, returns `(bestCost, gap, time)`.
- `show_stats.py -l <csv>` — mean gap and cost.
- Checkpoints are located relative to `model/` by the problem type:
  `model/{vrptw|cvrp}/checkpoints/checkpoint_epoch_*.pth` (moves) and
  `.../checkpointsE2E/...` (heat-map). The highest epoch number wins.
- Datasets: `MVMoE_data` (2004 VRPTW instances, 100 customers, normalised coords, capacity
  50), `MVMoE_data_50`, `NeuOpt_100`, `NeuOpt_50` (CVRP). Each has `BestObj.json`.

Runtime dependencies seen in imports: `torch`, `numpy`, `networkx`, `pyvrp`, `h5py`,
`torch_geometric` (imported in `layerGraphEmbedding.py` but not used at inference).

---

## 9. Mental model in one picture

```
                 ┌────────────── HistoryHH (≤32 basins) ──────────────┐
                 │ basin_0 (chain) … basin_k-1 (chain) │ basin_k (tree, ≤96) │
                 └──────────────────────────────────────┴─────────────────────┘
                                   │ sample() → one big Sample, blocks = basins
                                   ▼
             ┌── HeuristicNet ──┐            ┌── E2ENet ───────────────────┐
             │ node score P(v)  │            │ heat-map P_ij per solution │
             │ move score P(k|v)│            └──────────────┬─────────────┘
             └────────┬─────────┘                           │
     mask: old basins, │ tried pairs, uphill kids            │ best-in-basin or most-Hellinger-distant
                       ▼                                     ▼
       branch from v with heuristic k          greedy decode  /  3-route beam search
                       │                                     │
                       └────────────► new solution ◄─────────┘
                                          │
                                   history.update(...)
```

---

## 10. Where the learning signal comes from (inferred)

No training or data-generation code ships in this repo, but the label construction in
`model.py:118-128` and the `Sample` fields `g_dst`, `m_bst`, `t` imply an offline pipeline:
generate PSGs by running the 19 heuristics from many starting points, compute for every node
its graph distance to the basin's local optimum (`g_dst`) and the heuristic on the shortest
path (`m_bst`), and for the E2E model pair each solution with a target adjacency `t` (a
best-known / optimal solution). The models are therefore **imitation-learned**, not
RL-trained, at least as far as this code reveals.

---

## 11. Extending or debugging: what to touch

| want to… | look at |
|---|---|
| change how a solution is featurised | `historyHH.embed_sample`, `HistoryHH.update` (u[4..7]) |
| add a heuristic | `historyHH.applyHeuristics`, `ALNSLocal.applyHeuristics`, bump `PARAM_NEDGES_TYPE` (requires retraining: edge embedding table size) |
| change stagnation / jump policy | `HyperHeuristic.solve` lines 546–558 |
| change decoding of the moves model | `predictMove` masks and the `conditional_preditions` argmax |
| swap greedy jump for ruin-and-repair | call `predictdistroy` instead of `predictJump` |
| beam behaviour | `constrainedBeamHH.constrained_partial_beam_search_vrp` (route choice) and `partial_beam_search_vrp` (extension sampling) |
| memory bounds | `maxPSGsamples`, `maxPSGsolutions` in `HyperHeuristic.__init__` |

---

## 12. Quirks and discrepancies noticed

a. `HistoryHH.update` (`historyHH.py:273`): `del self[np.argmax(self.get_cost()) < self[0].n[0]]`
   evaluates to `del self[True]`, i.e. it always deletes element **1**, the second-oldest
   basin, regardless of cost. The comment says it preserves the best local minimum; in
   practice it preserves basin 0 and the newest basins.

b. *(fixed in this branch)* `RunHH.runHH` used to return `(bestCost, gap, time)` while
   `pRunHH.main_worker` unpacked the second value as `algNVehicles`, so the CSV column held the
   relative gap. It now returns the number of routes of the best solution; the master still
   recomputes `gap` itself from `algBestCost`. `RunHH.py` also gained CLI flags (`-d -i -t -g`)
   with defaults on a bundled dataset, replacing the hard-coded `MVMoE_data_5` path.

c. `intialPhase` records the ALNSLocal trajectory as a **chain** (`nId-1 → nId`), ignoring
   the actual parent `iterStartSol` stored in `ALNS.history`. The first PSG is thus a path
   even when ALNS rejected moves.

d. `drawSearchGraph` references `matplotlib`/`plt` without importing them; the NetworkX
   `searchGraph` mirror (`initialSG`/`updateSG`) is built in phase 1 but never updated in
   the main loop. The real state lives in `HistoryHH`.

e. `accept`, `temperature`, `nodeMoveFrequency`, `focusePeriod`, `lastIter2Global`, the
   `decode*` methods, `updateProbabilities`/`initWeights` in `HyperHeuristic` are legacy
   and not on the executed path.

f. `predictMove` uses `model_inference(..., device_ids=[self.device_id])` for PSG2 but the
   default `[0]` for PSG3; only PSG2 is used so this is harmless today.

g. The stopping criterion depends on the BKS (`gap ≤ 0.001`), so wall-clock results are not
   comparable to a method that does not know the BKS.

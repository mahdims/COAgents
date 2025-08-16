# COAgents

COAgents is a multi-agent framework that learns to orchestrate local improvement heuristics using search history. Three learned agents—Node Selection, Move Selection, and Jump—operate over a partial search graph.

## Requirements
- Python 3.11
- PyTorch 2.4

## Running Experiments

### Vehicle Routing with Time Windows (VRPTW)
```bash
python pRunHH.py -d ./dataset/MVMoE_data/ -l MVMoE_data.csv -t vrptw -m 4
python show_stats.py -l MVMoE_data.csv
```

### Capacitated Vehicle Routing (CVRP)
```bash
python pRunHH.py -d ./dataset/NeuOpt_100/ -l NeuOpt_100.csv -t cvrp -m 4
python show_stats.py -l NeuOpt_100.csv
```

Replace `-m` with the desired number of GPUs.


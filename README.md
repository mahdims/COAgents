**COAgents** is a general multi-agent framework that leverages search history to orchestrate local improvement heuristics via three learned agents—the Node Selection Agent (NSA), the *Move Selection Agent* (MSA), and the *Jump Agent* (JA)—operating over a partial search graph (PSG), which serves as the input to all agents during both training and inference.


**Requirements:**
Python3 v3.11
PyTorch v2.4

**Experiments**
To run VRPTW instances plesse use this command:
python3 pRunHH.py -d ./dataset/MVMoE_data/ -l ./MVMoE_data.csv -t 'vrptw' -m 4
where -m 4 means use 4 gpus for the run. After the run completed, the gap value can be extracted from the cvs file using command 
python3 show_stats.py -l ./MVMoE_data.csv

Similarly it can be run for CVRP instances as:
python3 pRunHH.py -d ./dataset/NeuOpt_100/ -l ./NeuOpt_100.csv -t 'cvrp' -m 4
and the gap can be summarized as
python3 show_stats.py -l ./NeuOpt_100.csv

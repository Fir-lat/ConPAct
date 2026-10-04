# ConPAct: Consistent Plan-Act for Long-Horizon Agentic Tasks

This repository provides an anonymized implementation of ConPAct for peer review, including inference-time reconciliation (ConPAct-I) and training-data construction from sampled interactions (ConPAct-S) and reconstructed trajectories (ConPAct-R).

- [ConPAct-I](src/conpact_agent/methods/conpact_i.py): inference-time reconciliation.
- [ConPAct-S](src/conpact_agent/data/conpact_s.py): supervision from sampled interactions.
- [ConPAct-R](src/conpact_agent/data/conpact_r.py): supervision from reconstructed trajectories.

## Run

Python 3.10+. Supply model and sandbox adapters through [configs/default.json](configs/default.json) using these [interfaces](src/conpact_agent/interfaces.py), and provide your task list as `tasks.json`.

```bash
pip install -e .
conpact-agent run --config configs/default.json --tasks tasks.json --out outputs/episodes.jsonl
conpact-agent metrics --input outputs/episodes.jsonl
```

## Train

Export training data with `conpact-agent data` (see `--help`). SFT uses ms-swift 4.x on NVIDIA GPUs.

```bash
pip install -e '.[sft]'
MODEL_PATH=./models/model DATASET=./data/train.jsonl OUTPUT_DIR=./outputs/sft \
  bash scripts/sft/joint.sh
```

For separate planner/actor training, use [separate.sh](scripts/sft/separate.sh).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Environments

Sokoban, Crafter, Procgen, MiniGrid, and OSWorld

Upstream environment repositories:

| Environment |  Upstream repository|
|-------------|-----|
| Sokoban |  https://github.com/lmgame-org/GamingAgent   |
| Crafter | https://github.com/danijar/crafter |
| Procgen | https://github.com/openai/procgen |
| MiniGrid | https://github.com/Farama-Foundation/Minigrid |
| OSWorld | https://github.com/xlang-ai/osworld |


## Citation

If you find this repo useful, please cite this work as follows:
```
@misc{li2026consistentplanactlonghorizonagentic,
      title={Consistent Plan-Act for Long-Horizon Agentic Tasks}, 
      author={Heng-Zhuang Li and Yi-Kai Zhang and Yu Wang and Yueqing Sun and Jiayuan Zhang and Qi Gu and Han-Jia Ye},
      year={2026},
      eprint={2609.38891},
      archivePrefix={arXiv},
}
```

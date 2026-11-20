# Reproduce

```bash
python -m pip install torch numpy tokenizers scikit-learn pytest
python reproduce/run_all.py   # every figure in the article, about half a minute on a CPU
python -m pytest -q           # the model, the trainer, the permissions and the contract, 119 tests
```

Python 3.10 or later, and no GPU. `run_all.py` evaluates the closed forms in [`src/finllm`](../src/finllm), reads
the Trial's measurements from [`evidence/trial.json`](../evidence/README.md), runs the contract checks on the
Trial model, compares each figure with the text, writes [`results.md`](results.md) and exits with status 1 if any
figure disagrees. The tests and `run_all.py` run on every push.

## What is recomputed, and from what

| Part of the article | How it is recomputed |
| --- | --- |
| §3 The price of unlicensed consistency | The fixed point of Equation (3.2) is solved by bisection in `finllm.permission_math` for each strength of the table; the implemented pair loss of `finllm.permissions` is then minimized over free logits and compared with it. The three-class case is checked the same way |
| §3 The response constraint | The calibrated response and the safe margin of Equation (3.3), and the hinge at a calibrated predictor and at a model that does not respond |
| §4 Units | Sibling probabilities for families of two, three and five views, the share of cells in which a sparse term is present, and the family and view weights of `finllm.objective` |
| §4 One update, one objective | The gradient of one update window on the Trial model three ways: as a single batch, as four ranks of two microbatches with global denominators, and as the same cells with per-cell averaging |
| §5 Projection | The loss and the gradients of projecting only the supervised positions against projecting every position, the size of both logits tensors at the two configurations, and the gap to a loss restricted to the label tokens |
| §5 Packing | Logits of three packed examples against their unpacked values, with segmented attention and with plain causal attention, and the gradient of one segment's loss with respect to an earlier segment's inputs |
| §6 The Trial | Token totals, tokens per parameter and retention differences from the stage records; every precision, recall, F1, macro-F1 and accuracy from the stored counts per class; the majority identity of Equation (6.1) on the development and test splits; the exchange between classes; the calibrated confidence |
| §7 Two scales | Both parameter counts from Equation (7.1), the per-layer and vocabulary parts, the training state in bytes, and the allocations of the recipe |

The contract checks build the Trial model from its configuration with a fixed random seed and run on the CPU.
They need no trained weights: an equivalence between two ways of computing one update is a property of the
computation and holds at any weights. Figures at the level of single-precision rounding are reported by order of
magnitude, since the last digits of a rounding error depend on the processor.

## The code

| Module | What it holds |
| --- | --- |
| [`finllm/permissions.py`](../src/finllm/permissions.py) | The four permissions, the license each one grants, the pair loss of Equation (2.1) and the scorecard |
| [`finllm/permission_math.py`](../src/finllm/permission_math.py) | The closed forms of §3: the optimum under a consistency term, its limit, the calibrated response and the safe margin |
| [`finllm/objective.py`](../src/finllm/objective.py) | Global denominators over an update window, the cell loss of Equation (4.2), per-cell averaging for comparison, and family and view weights |
| [`finllm/execution.py`](../src/finllm/execution.py) | Projection at supervised positions, the label-restricted loss for comparison, and packing with segmented causal attention |
| [`finllm/accounting.py`](../src/finllm/accounting.py) | Scores from a confusion table, the majority identity, the exchange between two models and sibling probabilities |
| [`finllm/architecture.py`](../src/finllm/architecture.py) | Parameters, training state and logits sizes of both configurations, and the recipe's allocations |
| [`finllm/model.py`](../src/finllm/model.py) · [`finllm/tokenizer.py`](../src/finllm/tokenizer.py) | The Trial's decoder and its byte-level BPE tokenizer |
| [`finllm/data.py`](../src/finllm/data.py) · [`finllm/corpus.py`](../src/finllm/corpus.py) · [`finllm/preparation.py`](../src/finllm/preparation.py) | Corpus packing with the task-text exclusion, splits by family, and the mixture of general and financial text |
| [`finllm/training.py`](../src/finllm/training.py) · [`finllm/checkpoint.py`](../src/finllm/checkpoint.py) | The trainer for language stages, stance supervision and preferences, with resumption and weight-only stage transitions |
| [`finllm/evaluation.py`](../src/finllm/evaluation.py) · [`finllm/inference.py`](../src/finllm/inference.py) · [`finllm/baselines.py`](../src/finllm/baselines.py) | Evaluation, calibration, the prediction interface and the TF-IDF baseline |

## Running the Trial's pipeline

The package installs a command, `finllm`, whose subcommands are the stages of the Trial in order.

```bash
python -m pip install -e .
finllm smoke --device cpu
```

`smoke` runs the whole path on a small synthetic set written by the package: tokenizer, language stage, stance
supervision, resumption, calibration and prediction. The Trial's own path is `freeze-exclusions`,
`train-tokenizer`, `pack`, `mix-corpora`, `train`, then `evaluate`, `calibrate` and `predict`.
[`configs/`](../configs) holds one configuration per training stage; each run of the Trial set its own budget of
updates, and the budgets are recorded in [`evidence/trial.json`](../evidence/trial.json). The corpora and the
reviewed headlines are kept outside the repository; their sources are listed in
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).

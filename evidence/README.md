# Evidence

The measurements of the 24M Trial. Every measured figure in the article is read from
[`trial.json`](trial.json) by [`reproduce/run_all.py`](../reproduce/run_all.py), which rederives each reported
score from the counts recorded here.

| Field | What it holds | Article |
| --- | --- | --- |
| `model` | The Trial's configuration: 6 layers, hidden size 512, 8 query heads over 4 key-value heads, a vocabulary of 8,192, a context of 512, tied embeddings | §6, §7 |
| `hardware` | One Tesla T4 with mixed precision, 4 sequences of 512 tokens per microbatch, 8 microbatches per update, and the peak of allocated memory in bytes | §6 |
| `tokenizer` | A byte-level BPE of 8,192 tokens trained on the Trial's own corpus | §6 |
| `exclusion` | The 10,686 distinct headlines excluded from the tokenizer's input and from every language corpus | §6 |
| `corpora` | Documents and tokens of each corpus: the seed, the expansion on general text with its validation part, the financial text, and the adaptation mix | §6 |
| `language_training` | The three stages: updates, supervised tokens, validation loss and the number of positions it was measured on, throughput, and the retention loss before and after each transfer | §6 |
| `review` | The review of 2,000 headlines: decisions confirmed, exclusions, unresolved cases, the three-class counts, the task pool, the split by family and the class counts of the training split | §6 |
| `sft` | The two stance runs. For each: epochs, optimizer updates, label positions, and on the development split the support, the predictions and the correct predictions per class. The baseline's two scores | §6 |
| `calibration` | The fitted temperature and the review threshold | §6 |

## Reading the fields

**Counts, then scores.** The stance results are stored as counts per class: support, predicted and correct.
Precision, recall, F1, macro-F1 and accuracy in the article are computed from those counts, so every reported
score can be traced to examples.

**Stages.** `seed`, `expansion` and `domain` are the three language stages in order. A stage that begins from an
earlier stage's weights records it under `initialization`: weights are transferred, and the optimizer, the
schedule and the data position start new.

**Retention.** `seed_retention_loss_before` and `seed_retention_loss_after` are the loss on the seed stage's
validation text before and after expansion. `general_retention_loss_before` and `general_retention_loss_after`
are the loss on general validation text before and after financial adaptation.

**Splits.** The split is by family: every view derived from one headline is in one partition. The calibration
split is used in two disjoint halves, one for the temperature and one for the review threshold.

**Labels.** The stance labels are my own review of each headline against its named target. The text of the
headlines and the review sheets are kept outside the repository.

## What is measured on the model directly

The contract measurements of §4 and §5 are not stored. They are run on the Trial model at every execution of
`run_all.py`, from [`reproduce/contract_checks.py`](../reproduce/contract_checks.py): the gradient of one update
under three reductions, the projection at supervised positions, and packing with segmented attention.

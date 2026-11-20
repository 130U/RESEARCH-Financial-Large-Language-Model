# Design notes

Why each piece of the design is the way it is, what it was chosen over, and how the Trial was run. Proofs are in
[derivations](derivations.md); figures are recomputed in [results](../reproduce/results.md).

## Choices and the alternatives they beat

| Choice | Alternative | Why | When I would switch |
| --- | --- | --- | --- |
| **A judgment relation as the unit of supervision** | A polarity label per sentence | One sentence can hold a rising metric and a bearish author; a polarity label has to pick, and the model cannot know which was picked (§1) | Text with one target, one holder and one fact, where polarity is the relation |
| **Separate axes for stance, metric direction and my own judgment** | One sentiment axis | "Revenue grew" and "the author is bullish" agree or diverge; a shared label teaches the model to merge them | No case |
| **Training from random initialization** | Fine-tuning a general model | The tokenizer and every corpus can be bounded, so the 10,686 task texts are excluded before any label exists, and the model's knowledge has a stated boundary | Tasks with no leakage concern and no need for a dated or bounded corpus |
| **Four permissions with two licenses** | One "label-preserving" flag per edit | Label preservation does not license distribution preservation: consistency on such a pair costs up to half the gap between the true posteriors ([D1](derivations.md#d1-the-price-of-unlicensed-consistency)) | Edit sets that contain only strict paraphrases |
| **A one-sided response term in log-odds** | A symmetric term on relabelling pairs, or none | Zero at the calibrated predictor for margins up to the edit's evidence weight, and equal to the margin for a model that does not respond ([D2](derivations.md#d2-the-relative-response-constraint-is-free-for-a-calibrated-model)) | No case |
| **Permissions per task axis** | One permission per edit | The same edit relabels one axis and is an equivalence on another (§2) | Single-axis models |
| **Permissions as metadata, never as input** | Edit names or tags in the prompt | A tag that names the relation is an answer cue; the model has to read the relation from the text | No case |
| **Global denominators over the update window** | Each cell averages over its own count | The gradient is then independent of ranks, microbatches and cell size; per-cell averaging changed it by 0.86 in relative norm on the Trial model ([D3](derivations.md#d3-one-update-one-objective), [D4](derivations.md#d4-what-per-cell-averaging-does)) | A single dense term on equal-sized cells, where the two coincide |
| **Families split before views are generated** | Splitting records | A held-out view of a three-view family has a sibling in training with probability 91.0% under record-level splitting ([D5](derivations.md#d5-families-views-and-siblings)) | No case |
| **Views averaged within a family** | Every view an independent example | Twenty edits of one text would otherwise weigh twenty times one unedited text | Terms defined on natural examples only, which keep their own distribution |
| **Projection at supervised positions with the full vocabulary** | Projecting every position, or a softmax over the label tokens | The first wastes a logits tensor 2,048 times larger; the second is a different loss, short by the log of the mass on the labels ([D6](derivations.md#d6-projecting-only-supervised-positions)) | Dense next-token training, where every position is supervised |
| **Segmented causal attention for packing** | Padding, or packing with plain causal attention | An example must not read its neighbor's answer; with rotary positions the packed logits equal the unpacked ones without resetting positions ([D7](derivations.md#d7-segmented-attention-on-a-rotary-model)) | Absolute position embeddings, which also need segment-local position ids |
| **Macro-F1 and per-class recall for selection** | Accuracy | A constant predictor scored 75.25% accuracy; Equation (6.1) is its signature ([D8](derivations.md#d8-a-constant-predictor-has-an-exact-signature)) | Balanced classes |
| **Exposure balanced in training, natural distribution in evaluation** | Reweighting the evaluation set | Sampling changes what the model sees; the development and test splits keep the class shares of the task | No case |
| **Calibration on examples used for nothing else, in two disjoint halves** | Calibrating on the development split | The temperature and the review threshold are each fitted once, and a temperature cannot change a predicted class ([D9](derivations.md#d9-temperature-cannot-repair-a-collapse)) | No case |
| **Stage transitions transfer weights only** | Carrying optimizer state into a new corpus | A new stage is a new optimization problem; resuming the same stage restores optimizer, random state and data position together | No case |

## One example

```text
text        the passage as the model receives it
target      the entity, instrument or metric the judgment is about
holder      whose view is asked for: the author, a quoted party, me
scope       the evidence the answer may rest on: the headline, the passage, the document
as_of       the time at which the information was available
task        stance | metric_direction | own_judgment
label       the answer on that task, or none
family      the source event this view derives from; fixed before any view is generated
```

A pair adds, for each task axis, a permission and, on a relabelling pair, the old and the new label. The public
Trial implements the stance task with a literal target and the headline as scope; the permission loss, the
scorecard and the contract's reductions are implemented and tested in the same package.

## The equivalence checks of the contract

Every change of execution has a check that runs on the public model. A change is admitted when its check passes.

| Change of execution | What must stay the same | Check | On the Trial model |
| --- | --- | --- | --- |
| Splitting an update over ranks and microbatches | The gradient of Equation (4.1) | Gradient from the split against the single-batch gradient | Equal to single-precision rounding; per-cell averaging differs by 0.86 |
| Projecting only supervised positions | The full-vocabulary loss and its gradients | Selected-position loss and gradient against the all-position ones | Loss equal; gradients equal to single-precision rounding |
| Packing several examples into one sequence | Each example's logits, and no information flow between examples | Packed logits against unpacked; gradient of one segment with respect to another | Within $`10^{-5}`$; gradient exactly 0 |
| Tying the embedding and the output head | One parameter object | Identity of the two tensors after construction and after loading | Holds |
| Resuming a run | The next update | The update after restoration against the update of an uninterrupted run | Covered by the trainer's tests |

The first three are in [`reproduce/contract_checks.py`](../reproduce/contract_checks.py) and run with every push.

## How the Trial was run

**Hardware and schedule.** One Tesla T4 with mixed precision; 4 sequences of 512 tokens per microbatch, 8
microbatches per update. The three language stages are separate optimization problems: the expansion and the
financial adaptation each start from the previous stage's weights with a new optimizer, schedule and data
position.

**Corpora.** General text is a prefix of one fixed English Wikipedia revision; financial text is the minutes of
the Federal Open Market Committee from 2011 to 2025, taken from the Board's official pages. The adaptation
mixture is 80% financial and 20% general by tokens, read once.

**Exclusion.** Every distinct headline of the SEntFiN pool, 10,686 texts, was excluded from the tokenizer's input
and from each language corpus, including headlines never sampled for review. Exclusion uses exact text, a
near-duplicate hash and source identity.

**Labels.** I reviewed 2,000 headlines, each with a named target, and decided investment stance on the headline
alone. The source's entity sentiment was not carried over: a report of a past price move did not become a stance
because its source label was positive or negative, and headlines about regulators or with unclear targets were
excluded where forcing them into neutral would have been the easy answer. The task pool keeps the examples whose
target appears literally in the headline.

**Splits.** By family: normalized text, near-duplicates and shared relationship identifiers stay in one
partition. The split is 705 training examples and 101 each for development, calibration and test, and the
calibration partition is divided once into 51 examples for the temperature and 50 for the review threshold.

**Selection and comparison.** Checkpoints were selected on development macro-F1. The first model, the rebalanced
model and the TF-IDF baseline are compared on that same split of 101 examples. The test split was read with the
first model, where it showed the same collapse. Between the first and the rebalanced run three things changed
together: exposure per class, the number of epochs and the update budget.

**The interface.** For an eligible input the model returns three probabilities, the class, the direction score
$`100\,(P_{\text{bullish}} - P_{\text{bearish}})`$ and a review flag set when the calibrated confidence is below
0.574. A score near zero can mean a confident neutral or two offsetting directions, so the probabilities are
always returned with it. An input that fails the eligibility checks returns no label, and is never reported as
neutral.

## What to monitor

| Quantity | What it tells | In the Trial |
| --- | --- | --- |
| Distance from the majority identity of Equation (6.1) | Whether a score carries information about the task | First model on the curve; rebalanced model 0.3687 above it |
| Per-class recall beside accuracy | What a gain in one class cost in another | 17 directional examples gained, 14 neutral given up |
| Invariance gap on equivalent pairs | Whether the model moves where nothing changed | Reported by the scorecard |
| Response on relabelling pairs | Whether the model moves where the answer changed | Reported by the scorecard |
| Retention loss on the previous stage's text | What a stage cost on what came before | 3.984 to 4.057 on general text |
| Gradient difference between a split and a single batch | Whether the runtime still trains the recipe | Single-precision rounding |
| Fitted temperature | Over- or underconfidence of the raw model | 1.506 |

## Next

1. **A dated, family-disjoint holdout.** New headlines after the training texts, grouped by event, read once with
   the selected model.
2. **Exposure against budget.** The rebalanced run at the first run's update budget, and the first run at the
   rebalanced one's, to separate the two changes.
3. **Reviewed edit pairs at scale.** Pairs with permissions on the stance and metric-direction axes, the pair
   loss against cross-entropy on the same pairs, and the scorecard on held-out pairs.
4. **The metric-direction axis in the public Trial.** The second task prefix, sharing the backbone with stance.
5. **Preferences.** The trainer has a preference stage; pairs derived from reviewed labels are its first input.

# Financial Large Language Model: Semantic Permissions and Objective-Preserving Scaling

**Independent research**  
Letao (Theodore) Ouyang · November 2026

**Materials.** The model exists at two scales: a 2.4B-parameter version that I use for my own research, and a 24M-parameter Trial that is public in this repository. Every measured figure below comes from the Trial, which was trained from random initialization on a single Tesla T4; its records are in [`evidence/trial.json`](../evidence/trial.json). Figures for the 2.4B configuration are exact arithmetic on its configuration and its training recipe. The contract measurements of Sections 4 and 5 are run on the Trial model by [`reproduce/run_all.py`](../reproduce/run_all.py), which recomputes every figure in this article. The stance comparisons of Section 6 are on the Trial's 101-example development split, the split that also selected the checkpoint.

## Summary

Financial text packs three things into one sentence: a change in a fact, the author's attitude, and an action. A sentiment label compresses them into positive or negative, and a model trained on it learns to detect cue words while answering the wrong question. I built a financial language model around a different object, a judgment relation: who expresses which view of what, on which evidence, as of when, and what would change it. Two designs carry it. **Financial semantic permissions** state, for every edit of a text and every task axis, what the model's output may and must do, and keep two licenses apart that consistency training merges: preserving the label and preserving the distribution. I prove that the merge has a price, a calibration error of up to half the distance between the two true posteriors, and that a relative-response constraint on log-odds has none. **Objective-preserving scaling** is a training contract under which an update has one objective however it is executed: split over devices, accumulated over microbatches, packed, or projected only where it is supervised. On the public model the gradient under the contract agrees across splits to one part in a million, where per-cell averaging changes it by 86%.

## Problem and Approach

I wanted a model that reads financial text the way I do when I work: this author is bearish on this company, on this evidence, as of this date, and would change their mind if that happened. The standard supervision cannot express that sentence. A polarity label has no target, no holder and no date, and it is assigned by consensus, where my use needs one analyst's standard applied consistently. So I took control at the level of training: how material enters the model, where supervision applies, how tasks share a representation, and what an execution change is allowed to alter.

Training from random initialization is what makes that control complete. The tokenizer and every corpus are mine to bound, so the texts I later label are excluded from both before any label exists, and the model's knowledge has a boundary I can state. I validated the whole path at 24M parameters, where each adjustment can be tied to specific examples and counts, and carried the task protocol and the training contract to 2.4B.

The Trial's most instructive result came from its first stance model, which predicted neutral for every example and still scored 75.25% accuracy. That number is exactly the score of a constant predictor, which is how I knew what it meant. Rebalancing exposure recovered the directional classes and cost neutral recall, and the exchange is itemized below. The principle I kept: evaluate what the model learns and the errors it introduces to learn it.

## Key Contributions

* **I replaced the sentiment label with a judgment relation.** Every example names a target, an opinion holder, an evidence scope and an as-of time, and investment stance, metric direction and my own judgment are separate task axes on a shared backbone.
* **I defined financial semantic permissions.** For each edit and each axis one of four permissions applies: equivalent, label-preserving, relabel, unconstrained. Label preservation and distribution preservation are separate licenses, and the same edit can carry different permissions on different axes.
* **I proved what merging the two licenses costs.** A consistency term applied to a pair whose true posteriors are $`p`$ and $`p'`$ moves both predictions toward each other by the same amount, up to $`\left\vert p' - p\right\vert/2`$: two calibrated posteriors of 0.70 and 0.90 are both driven toward 0.80 (Equation 3.2).
* **I showed that the relative-response constraint is free for a calibrated model.** A hinge on the change in log-odds of the new label against the old one is zero at the calibrated predictor for every margin up to the edit's evidence weight; with labels assigned at posterior odds of at least 2 to 1, any margin up to $`2\ln\,2 = 1.39`$ is safe (Equation 3.3).
* **I wrote the training contract and measured it.** With global denominators the gradient of an update is the same under every split into ranks and microbatches: on the Trial model the relative difference is of order $`10^{-6}`$, the rounding of single precision, against 0.86 for per-cell averaging (Equation 4.2).
* **I priced the output projection.** Projecting only supervised positions leaves the full-vocabulary loss and its gradients unchanged and shrinks the logits tensor by the ratio of positions: from 1 GiB to 0.5 MiB at the 2.4B configuration's microbatch (Equation 5.1). Restricting the loss to the label tokens is a different objective, and the difference is the log of the probability mass on those tokens.
* **I verified packing on a rotary model.** With segmented causal attention a packed example receives the logits it would receive alone, to a few parts in a million, and the gradient across a segment boundary is exactly zero; rotary positions are relative, so no position is reset.
* **I trained the Trial end to end and diagnosed its collapse exactly.** 26,115,096 supervised tokens over three language stages on one T4, then stance supervision on headlines I reviewed. The first model's 75.25% accuracy and 0.2863 macro-F1 are the majority-class identity $`\tfrac{1}{K}\cdot\tfrac{2\pi}{1+\pi}`$ to four digits; the rebalanced model reaches 0.6549, above a TF-IDF baseline's 0.6087, by gaining 17 directional examples and giving up 14 neutral ones.

## Manuscript

### 1. A sentiment label answers a different question

Financial sentiment analysis has a standard form. A sentence receives one of three polarity labels, from the Financial PhraseBank (Malo et al., 2014) onward, and models are compared on agreement with those labels: domain-pretrained encoders (Araci, 2019; Yang, Uy and Huang, 2020), then instruction-tuned generative models and the benchmarks built around them. The form has been refined in two directions. Entity-level corpora attach the label to a named target: SEntFiN annotates 10,753 headlines, 2,847 of which contain several entities, "often with conflicting sentiments" (Sinha et al., 2022). And stance corpora change what is labelled: TweetFinSent labels whether a tweet's author expects a gain or a loss on a stock (Pei et al., 2022), and stance toward financial metrics has been annotated in filings and earnings calls (Gyawali et al., 2025).

What a polarity label leaves out is easiest to see in one sentence:

> Company A's revenue grew year over year, but I am still reducing my position in A.

The metric moved up. The author's investment stance is bearish. If the reduction reflects a portfolio constraint, the stance toward the company may be neither. A single label has to pick one of these, and a model trained on it has no way to know which was picked.

The consequence is measurable. An attack that inverts the financial meaning of a sentence through targeted lexical substitutions deceives FinBERT in 90.85% of eligible cases, the model frequently keeping its original prediction (Cioboata and Bildea, 2026); on aspect-level sentiment in general, test sets that disentangle the target aspect from the others reduce the accuracy of nine models by up to 69.73% (Xing et al., 2020). The models respond to words where they should respond to relations.

Opinion mining stated the relation long ago. Liu (2012) defines an opinion as a quintuple of entity, aspect, sentiment, holder and time, and holder extraction is an old task (Kim and Hovy, 2006; Wiebe, Wilson and Cardie, 2005). I use that structure as the unit of supervision. Each example carries

```math
\big(\text{text},\ \underbrace{\text{target}}_{\text{about what}},\ \underbrace{\text{holder}}_{\text{whose view}},\ \underbrace{\text{scope}}_{\text{on which evidence}},\ \underbrace{t_{\text{as of}}}_{\text{known when}}\big) \;\longmapsto\; y_j \quad \text{for task axis } j , \qquad (1.1)
```

with three axes: the **investment stance** a specified holder expresses, the **metric direction** relative to an explicit comparator, and **my own judgment**, which adds the information and constraints I had at the time. The axes share a backbone and a vocabulary head and are selected by the task prefix. Because the labels encode one analyst's standard, they are a different object from a consensus label; the literature on label variation argues that the consensus was never the only truth (Aroyo and Welty, 2015; Plank, 2022).

### 2. Permissions: what an edit may and must do

A label on one text constrains one output. Financial reading is mostly about differences: the same sentence with another target, a quotation where there was an assertion, a negation, a condition added. I supervise the differences directly. An **edit** maps a text to an edited text, and for each task axis the pair carries one **permission**, assigned when the pair is reviewed.

**Table 1. The four permissions an edit can carry.**

| Permission | The label | The distribution | Loss on the pair | Example on the stance axis |
| --- | --- | --- | --- | --- |
| Equivalent | preserved | preserved | consistency | "shares" for "stock"; a reordered clause |
| Label-preserving | preserved | free to move | none | a second reason for the same view; a hedge taken out |
| Relabel $`a \to b`$ | changes | changes | relative response | the target swapped; "not" inserted; the holder changed from a quoted analyst to the author |
| Unconstrained | not related | not related | none | a rewrite whose effect on the view cannot be determined |

Every labelled endpoint keeps its cross-entropy anchor. On a pair $`(x, x')`$ with class logits $`z, z'`$ and labels $`y, y'`$ the loss is

```math
\ell(x, x') \;=\; \mathrm{CE}(z, y) + \mathrm{CE}(z', y') \;+\; \lambda_E\,𝟏[\text{equivalent}]\ \big\lVert \bar z - \bar z' \big\rVert^2 \;+\; \lambda_R\,𝟏[\text{relabel}]\ \max\big(0,\ m - D\big), \qquad (2.1)
```

where $`\bar z`$ is $`z`$ with its mean removed, so that the consistency term compares log-odds, and

```math
D \;=\; \big(z'_b - z'_a\big) - \big(z_b - z_a\big) \qquad (2.2)
```

is the change across the edit in the log-odds of the new label against the old one. Permissions, labels and edit names are supervision metadata. The model's input holds the text, the task and the target, and nothing that names the answer.

Two properties make this more than a relabelling of known techniques. First, **label preservation and distribution preservation are separate licenses**. A paraphrase preserves both. A second reason for the same view preserves the label and should raise the confidence. Consistency training constrains predictions to be invariant to input noise and augmentation (Xie et al., 2019), which treats a label-preserving transformation as distribution-preserving; Section 3 shows what that costs. Second, **a permission belongs to an axis**. The same edit is read differently by different tasks:

**Table 2. Permissions of four edits of the sentence in Section 1 on two axes.**

| Edit of the sentence in Section 1 | Stance axis | Metric-direction axis |
| --- | --- | --- |
| "grew" becomes "fell" | label-preserving: the author is still reducing | relabel: up to down |
| "I am still reducing" becomes "I am adding to" | relabel: bearish to bullish | equivalent |
| "Company A" becomes "Company B" in the second clause only | relabel for target A: bearish to neutral | equivalent for target A |
| "I" becomes "one analyst, I am told," | relabel for the author as holder | equivalent |

A single invariance or sensitivity flag per edit cannot represent this table. The permission matrix says where the tasks may support each other and where their information must stay apart.

Behavioral testing already distinguishes invariance tests from directional expectation tests (Ribeiro et al., 2020), contrast sets and counterfactually revised data supply label-changing edits (Gardner et al., 2020; Kaushik, Hovy and Lipton, 2019), and CEBaB annotates aspect-level edits (Abraham et al., 2022). Those are instruments of evaluation and of data collection. Permissions turn the same distinctions into terms of the training objective, with the distribution license made explicit and the response measured in log-odds. The scorecard of the public code reads the same metadata at test time: the invariance gap on equivalent pairs, the share of labels kept, and the share of relabelling pairs on which the model responded.

### 3. Label-preserving is not distribution-preserving

Take one class against the rest, and a pair whose true posteriors for that class are $`p`$ at the original and $`p'`$ at the edited text. Let the model's predictions be $`q`$ and $`q'`$, and apply a consistency term of strength $`\lambda`$ to their log-odds. The population objective for the pair is

```math
J(q, q') \;=\; \mathrm{CE}(p; q) + \mathrm{CE}(p'; q') + \frac{\lambda}{2}\,\big(\mathrm{logit}\,q - \mathrm{logit}\,q'\big)^2 . \qquad (3.1)
```

**The price of unlicensed consistency.** At the minimum of $`J`$,

```math
q - p \;=\; p' - q' \;=\; \lambda\,\big(\mathrm{logit}\,q' - \mathrm{logit}\,q\big) \;=:\; \varepsilon(\lambda), \qquad 0 < \varepsilon(\lambda) < \frac{\left\vert p' - p\right\vert}{2}, \qquad \varepsilon(\lambda) \uparrow \frac{\left\vert p' - p\right\vert}{2} \ \text{as } \lambda \to \infty . \qquad (3.2)
```

The two predictions are pulled toward each other by the same amount, the original overconfident and the edited underconfident when $`p < p'`$, and the amount grows with the strength up to half the gap. If the edit is a true equivalence, $`p = p'`$ and the cost is zero for every $`\lambda`$. The statement extends to $`K`$ classes: the error vectors at the two ends are equal and opposite, and in the limit both predictions equal the average of the two posteriors.

A worked case. An edit adds a second reason for a bullish view, and a calibrated reader moves from 0.70 to 0.90. The evidence the edit carries is the change in log-odds, 1.35 nats, a likelihood ratio of 3.86.

**Table 3. Predictions and calibration error as consistency strength grows.**

| Consistency strength $`\lambda`$ | 0 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | $`\infty`$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prediction at the original | 0.700 | 0.728 | 0.756 | 0.779 | 0.793 | 0.797 | 0.799 | 0.800 |
| Prediction at the edited text | 0.900 | 0.872 | 0.844 | 0.821 | 0.807 | 0.803 | 0.801 | 0.800 |
| Calibration error at each end | 0 | 0.028 | 0.056 | 0.079 | 0.093 | 0.097 | 0.099 | 0.100 |

At a strength of 1 the model reports 0.79 and 0.81 for two texts that warrant 0.70 and 0.90: the consistency term has erased almost all of the evidence the edit added. Minimizing the implemented loss over free logits reaches the same numbers.

**The relative-response constraint is free for a calibrated model.** On a relabelling edit the calibrated predictor has

```math
D^{*} \;=\; \log\,\frac{p'_b}{p'_a} - \log\,\frac{p_b}{p_a} \;\ge\; 2\log\,r \qquad \text{when both labels were assigned at posterior odds of at least } r , \qquad (3.3)
```

so the hinge $`\max(0, m - D)`$ vanishes at the calibrated predictor for every margin $`m \le D^{*}`$. With $`r = 2`$ the safe margin is $`2\ln\,2 = 1.39`$ nats. A model that does not respond, $`D = 0`$, pays the full margin. The constraint therefore separates two things that a consistency term confuses: it penalizes the absence of a response and leaves the size of a correct response to the anchors.

The contrast is the reason for the license split. A symmetric penalty on a pair is safe only when the pair's posteriors are equal; a one-sided penalty on the direction of change is safe whenever the direction is known. Vision has reached the neighboring conclusion from the data side: soft augmentation lowers the target's confidence as the transform grows more aggressive and reports better calibration (Liu et al., 2022). The theorem states the cost on the loss side, for any task in which label-preserving edits carry evidence, and finance is such a task almost by definition.

### 4. One update, one objective

A hard failure of scaling is the silent one: the code runs, the loss falls, and the model is optimizing a different objective. The streams of this model make that easy. Running text supplies a supervised position at almost every token, a question-answer pair supplies a few, and a stance example supplies one. Twenty edits of one source text are not twenty independent events. And an optimizer update is assembled from pieces: ranks, accumulation microbatches, packed sequences.

**Units.** A **family** is everything derived from one source event: the original text, its target queries, its question-answer pairs and its edits. A **view** is one member of a family. A **position** is one supervised token. Families are split into partitions before any view is generated. Under independent assignment of views at the Trial's training share of 705 in 1,008, a held-out view would find a sibling in training with probability 69.9% for families of two views, 91.0% for three and 99.2% for five, which is why the unit of splitting is the family. Within a term, views are averaged inside a family before families are averaged, so a family with twenty edits weighs as much as a family with none.

**The objective of an update.** Let the window be all examples consumed between two optimizer steps, over every rank and every microbatch. For term $`j`$ with weight $`\lambda_j`$, loss sum $`S_j`$ and effective count $`C_j`$ over the window,

```math
L \;=\; \sum_{j:\, C_j > 0} \lambda_j\, \frac{S_j}{C_j} . \qquad (4.1)
```

Split the window into cells, one per rank $`r`$ and microbatch $`k`$, with term sums $`S_{rk,j}`$. Let each cell contribute

```math
\ell_{rk} \;=\; W \sum_{j:\, C_j > 0} \lambda_j\, \frac{S_{rk,j}}{C_j}, \qquad \frac{1}{W}\sum_{r}\sum_{k} \nabla \ell_{rk} \;=\; \nabla L , \qquad (4.2)
```

with gradients summed over microbatches and averaged over the $`W`$ ranks. The identity holds for every partition of the window, by linearity. The gradient is a function of the recipe and the data, and of nothing about the hardware.

The common alternative averages each term over the cell's own count. An example of term $`j`$ in a cell that holds $`n`$ of them then weighs $`\lambda_j / (N_{\text{cells}}\, n)`$ where the recipe says $`\lambda_j / C_j`$, and a term present in a share $`\pi_j`$ of the cells keeps a share $`\pi_j`$ of its weight. Goyal et al. (2017) gave the single-term rule for distributed training: normalize the per-worker loss by the total minibatch size, not the per-worker size. With several terms of unequal density the error is larger and depends on the layout. The recipe of the 2.4B model gives stance and event inputs 1% of the adaptation tokens; a term carried by 1% of the sequences is present in 1.99% of cells of two sequences, 7.7% of cells of eight and 27.5% of cells of thirty-two, so under per-cell averaging its weight would be set by the cell size.

**Measured on the Trial model.** A window of 16 sequences, twelve of running text, two question-answer pairs and two stance examples, gives 756 language positions, 16 answer positions and 2 stance positions. I computed the gradient three ways: the window as one batch; four ranks of two microbatches under Equation (4.2); and the same eight cells with per-cell averaging.

**Table 4. Difference of two gradient reductions from the single-batch gradient on the Trial model.**

| Reduction | Relative difference from the single-batch gradient |
| --- | ---: |
| Global denominators, 4 ranks × 2 microbatches | $`\approx 10^{-6}`$ |
| Per-cell averaging, same cells | 0.86 |

The first number is single-precision rounding. The second is a different objective: the stance term sits in one cell of eight and keeps an eighth of its weight, and the answer term keeps a quarter.

Mixing rates and task weights have their own literature (Raffel et al., 2019; Kendall, Gal and Cipolla, 2017). The contract is prior to it. Whatever weights are chosen, Equation (4.2) is what makes them the weights that are trained.

### 5. Execution that preserves the objective

Each optimization of the runtime changes how the update is computed and is required to leave Equation (4.1) and the permitted information flow unchanged. Each has an equivalence check that runs on the public model.

**Projecting only supervised positions.** A stance example supervises one position, the one that predicts the answer. The backbone still processes the whole context; the vocabulary head is applied to the selected hidden states only:

```math
H = \text{backbone}(x) \in ℝ^{B \times S \times d}, \qquad \text{logits} = H[\text{supervised}]\, W_{\text{vocab}}^{\top} \in ℝ^{M \times V}, \qquad \text{loss} = \mathrm{CE}_{\text{full vocabulary}} . \qquad (5.1)
```

The loss and its gradients are those of projecting every position and then selecting. On the Trial model the two losses are equal and the gradients differ by about $`10^{-6}`$ in relative norm. What changes is the tensor: at the 2.4B configuration a microbatch of 2 sequences of 2,048 tokens over a vocabulary of 65,536 gives a single-precision logits tensor of 1 GiB, and its two supervised positions need 0.5 MiB, a factor of 2,048. At the Trial's batch of 4 sequences of 512 tokens the figures are 64 MiB and 128 KiB. The saving in computation is the head's share of a position's matrix work, 7.0% at 2.4B and 17.3% at 24M, where the vocabulary matrix is a larger part of a smaller model.

Cut cross-entropy (Wijmans et al., 2024) removes the logits matrix for dense next-token training by computing the correct-token logit and the log-sum-exp on the fly. Sparse supervision needs less machinery, because most positions carry no loss at all. What it must not do is restrict the softmax to the label tokens. That is a different objective: the full-vocabulary loss exceeds the label-restricted one by exactly $`-\log`$ of the probability mass the model puts on the label tokens, 15.66 nats over two positions and three label tokens at initialization, and the difference is the part of the gradient that teaches the model to answer with a label at all.

**Packing with segmented attention.** Short examples are packed into one sequence to fill the context. Attention is causal within a segment and blocked across segments,

```math
\text{allowed}(i, j) \;=\; 𝟏\big[\mathrm{seg}(i) = \mathrm{seg}(j)\big]\cdot 𝟏[\,j \le i\,] , \qquad (5.2)
```

so one example cannot read its neighbor's answer, and different views of one family stay in separate segments. With rotary position embeddings the attention score depends on $`i - j`$ alone (Su et al., 2021), so a packed segment sees the scores it would see standing alone and no position needs to be reset. On the Trial model, three examples packed into 64 positions receive logits within $`10^{-5}`$ of their unpacked values, against differences of 2.9 under plain causal attention, where logits are of typical magnitude 0.36; the gradient of one segment's loss with respect to an earlier segment's inputs is exactly zero with the mask and 0.88 without it. Krell et al. (2021) introduced packing without cross-contamination for encoders; the check above is the decoder's version.

**State.** The input embedding and the output head are one parameter object in both configurations (Press and Wolf, 2016). The 2.4B model's weights, gradients and two Adam moments in single precision occupy 38.36 GB before activations, so precision, sharding and recomputation are part of the runtime design, and a resumed run restores optimizer state, random state and the data position together. A transition to a new stage transfers weights only and starts new optimization state, which the Trial did twice.

### 6. The Trial: from random initialization to a calibrated stance model

**The model and its boundary.** A 6-layer dense decoder of 24,255,744 parameters with grouped-query attention, rotary positions, RMSNorm, QK normalization, SwiGLU and tied vocabulary weights, adapted from the MiniMind architecture, with a byte-level BPE tokenizer of 8,192 tokens that I trained on the Trial's own corpus. Before the tokenizer saw any text, all 10,686 distinct headlines of the pool I would later label were excluded from the tokenizer's input and from every language corpus. The exclusion costs nothing when a model is trained from scratch and cannot be had afterwards, which is the same reason point-in-time models are trained from scratch (He et al., 2025a; Yan et al., 2026; Kelly et al., 2026).

**Language training.** General text is a prefix of one fixed English Wikipedia revision. Financial text is the minutes of the Federal Open Market Committee from 2011 to 2025. Training ran on one Tesla T4 with mixed precision, 4 sequences of 512 tokens per microbatch and 8 microbatches per update, at about 3,200 to 3,500 tokens per second and a peak of 1.06 GiB of allocated memory.

**Table 5. The language-training stages of the Trial.**

| Stage | Updates | Supervised tokens | Validation loss | Retention |
| --- | ---: | ---: | ---: | --- |
| Seed | 300 | 4,914,941 | 5.542 | |
| Expansion on new general text, weights transferred | 1,220 | 19,988,195 | 3.693 | Loss on the seed's validation text 4.403 → 3.984 |
| Financial adaptation: 80% financial, 20% general | 74 | 1,211,960 | 3.238 | Loss on general text 3.984 → 4.057 |
| **Total** | | **26,115,096** | | |

The three validation losses are on three different texts and are not a single curve. The retention column is the comparable one: expansion improved the model on text it had left behind, and financial adaptation cost 0.073 nats on general text while fitting financial text to 3.238. The total is 1.08 supervised tokens per parameter.

**Labels.** I reviewed 2,000 headlines from SEntFiN, each with a named target. Entity sentiment and investment stance are different labels, so every source label was reconsidered: 1,993 decisions were confirmed, 114 of them exclusions, and 7 were left unresolved, giving 1,879 three-class examples, 1,219 neutral, 426 bullish and 234 bearish. Requiring the target to appear literally in the headline left a task pool of 1,008, split by family into 705 for training and 101 each for development, calibration and test.

**A collapse with an exact signature.** The first stance model, 2 epochs and 45 updates from the adapted weights, predicted neutral for all 101 development examples and all 101 test examples. For a predictor that always answers the largest of $`K`$ classes, whose share is $`\pi`$,

```math
\text{accuracy} = \pi, \qquad \text{macro-F1} = \frac{1}{K}\cdot\frac{2\pi}{1 + \pi} . \qquad (6.1)
```

On development $`\pi = 76/101`$: accuracy 75.25% and macro-F1 0.2863. On test $`\pi = 59/101`$: 58.42% and 0.2458. Those are the reported scores, to the last digit. A score that matches Equation (6.1) carries no information about the task, however respectable the accuracy looks. The cause was arithmetic too. The training split is 491 neutral, 140 bullish and 74 bearish, a predictor that outputs those shares already has a cross-entropy of 0.81 nats, and 45 updates were enough to fit the shares and not enough to learn features that beat them.

**Rebalancing, and what it exchanged.** The second run sampled every class to 491 exposures per epoch from the same 705 training examples, for 3 epochs, 139 updates and 4,419 label positions.

**Table 6. Development-set metrics by class after rebalancing.**

| Development, 101 examples | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Bearish | 10 | 0.4545 | 0.5000 | 0.4762 |
| Neutral | 76 | 0.9394 | 0.8158 | 0.8732 |
| Bullish | 15 | 0.5000 | 0.8000 | 0.6154 |
| **Macro-F1** | | | | **0.6549** |

**Table 7. The rebalanced model against the first model and a TF-IDF baseline.**

| | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| First model, all neutral | 75.25% | 0.2863 |
| Class-weighted logistic regression on TF-IDF | 75.25% | 0.6087 |
| Rebalanced model | 78.22% | 0.6549 |

The rebalanced model classified 17 more directional examples correctly and 14 fewer neutral ones, 79 of 101 against 76. Accuracy moved by three examples and macro-F1 by 0.3687. The exchange is the result: direction was bought with neutral recall at a rate of 17 to 14, and whether that rate is acceptable is a decision about use that the single number hides. A linear model on word features reaches 0.6087 on the same split, so 0.0462 of the 0.6549 is what the pretrained decoder adds at this data size. Exposure, epochs and the update budget changed together between the two runs.

**Calibration.** Temperature and the review threshold were fitted on disjoint halves of the calibration split, 51 and 50 examples. The fitted temperature is 1.506: the model was overconfident, and a raw two-class confidence of 0.90 becomes 0.81. Predictions below a confidence of 0.574 are sent to review. A positive temperature never changes the predicted class, so calibration could not have repaired the collapse; the two problems are separate and were handled separately. The released interface returns the three probabilities, the class, a direction score $`100\,(P_{\text{bullish}} - P_{\text{bearish}})`$ and a review flag, and an input that fails the eligibility checks returns no label.

### 7. Two scales, one contract

**Table 8. The 24M Trial and the 2.4B configuration under one contract.**

| | 24M Trial | 2.4B |
| --- | ---: | ---: |
| Parameters | 24,255,744 | 2,397,459,968 |
| Layers | 6 | 28 |
| Hidden size | 512 | 2,560 |
| Feed-forward size | 1,664 | 8,320 |
| Query heads / key-value heads | 8 / 4 | 20 / 4 |
| Head dimension | 64 | 128 |
| Vocabulary / context | 8,192 / 512 | 65,536 / 4,096 |
| Parameters per layer | 3,343,488 | 79,631,616 |

Both counts follow from one formula. A layer holds query and output projections of $`d \times d`$, key and value projections of $`d \times (g\,h)`$ for $`g`$ key-value heads of dimension $`h`$, three feed-forward matrices of $`d \times f`$, two RMSNorm scales and two QK-norm scales:

```math
P \;=\; L\,\big(2d^2 + 2d\,g\,h + 3d\,f + 2d + 2h\big) + V d + d . \qquad (7.1)
```

For the 2.4B configuration a layer has 79,631,616 parameters, 28 layers have 2,229,685,248, the shared vocabulary matrix adds 167,772,160 and the final norm 2,560.

The 2.4B recipe allocates 50B input tokens to the language foundation and 5B to financial adaptation: 4B of financial source text, 0.75B of general text, 0.20B of reading questions and answers and 0.05B of stance and event tasks. That is 20.9 foundation tokens per parameter, on the line of scaling tokens with parameters (Hoffmann et al., 2022). The human-labelled allocation is 32,000 records: 12,000 for training, 4,000 for development, 4,000 for calibration, 4,000 for threshold selection and 8,000 for final test. These are allocations of the recipe.

What carries from 24M to 2.4B is the task protocol, the permissions, the contract of Sections 4 and 5 and the error analysis of Section 6. What is rebuilt is everything bound to a scale: the vocabulary, the weights, the context handling, the training state and the calibration, which belongs to one set of weights, one tokenizer and one target mode.

Financial models of similar and larger size are built for other questions. BloombergGPT is a 50B-parameter model trained on 363 billion tokens of financial data with 345 billion of general text (Wu et al., 2023). Point-in-time models bound what the model has read by date: twelve 1.3B models on about 100 billion tokens each (Yan et al., 2026), and decoders of up to 4B parameters on a trillion tokens (Kelly et al., 2026). Eva-4B classifies evasion in earnings-call answers and is fine-tuned from a general 4B model (Ma, Lin and Yang, 2026); FinanceParam adapts a 2.9B model to financial knowledge (BharatGen, 2025); FinLCR targets long-context financial reasoning (Qiu et al., 2026). This model bounds a different thing. It bounds what the model is asked to judge, to a relation with a target, a holder, a scope and a date, and what an edit is allowed to do to the answer, and it holds the objective fixed while the execution scales by a factor of one hundred.

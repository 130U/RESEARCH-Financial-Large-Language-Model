# Where this work sits

What should a financial language model be supervised to say, and what must stay fixed while its training scales?
Financial sentiment analysis supplies the task and its benchmarks. Behavioral testing and counterfactual data
supply edits whose effect on the answer is known. Consistency training turns invariance into a loss. Distributed
training, packing and loss computation each have their own engineering literature, and point-in-time models show
why a model is trained from scratch. Each holds part of the answer. This page records what each line establishes,
where it stops, and what the article adds. The complete list of 150 sources is in the
[bibliography](bibliography.md).

## The short version

| Line of work | What it establishes | Where it stops | What the article adds |
| --- | --- | --- | --- |
| Financial sentiment and stance: Financial PhraseBank, FinBERT, SEntFiN, TweetFinSent | Domain text needs domain models; sentiment can be attached to an entity; stock stance differs from tone | One label per text or per entity, by consensus, with no holder, scope or date | A judgment relation as the unit, with stance, metric direction and my own judgment as separate axes |
| Behavioral tests and counterfactual edits: CheckList, ARTS, contrast sets, counterfactually augmented data | Models respond to cue words; edits with a known effect expose it | Instruments of evaluation and data collection | The same distinctions as terms of the training objective, per task axis |
| Consistency training: UDA, R-Drop | Invariance to label-preserving noise is a strong regularizer | Label preservation is taken to license distribution preservation | The two licenses separated, with the calibration cost of merging them in closed form |
| Calibration: Guo et al., soft augmentation | Modern networks are miscalibrated; targets should soften as inputs degrade | Fixed after training, or on the data side | A response constraint that is provably free for a calibrated model |
| Distributed training, packing, large-vocabulary loss: Goyal et al., Krell et al., Wijmans et al. | Normalize by the total minibatch; pack without cross-contamination; avoid materializing logits | One technique at a time, one loss term | A contract over several terms of unequal density, with an equivalence check for each change of execution |
| From-scratch and point-in-time models: PubMedBERT, ChronoGPT, DatedGPT, Kelly et al. | From-scratch pretraining wins where domain text is abundant, and is the only way to bound what a model has read | The boundary is on the corpus | The same control applied to the task: texts to be labelled are excluded before the tokenizer exists |
| Imbalance and metrics: King and Zeng, logit adjustment, macro-F1 | Majority classes dominate; accuracy hides it | General methods | An exact signature that identifies a constant predictor from its two scores |

## Key sources

### Financial sentiment and its labels

**Malo et al. (2014), "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts."** [doi](https://doi.org/10.1002/asi.23062)
The Financial PhraseBank: a human-annotated phrase bank for training and evaluating models of financial polarity,
built on the observation that the orientation of a sentence can differ from that of its words. It fixed the form
of the task for a decade: one sentence, one of three labels. The form has no slot for whose view the label is or
what it is about.

**Loughran and McDonald (2011), "When Is a Liability Not a Liability?"** [doi](https://doi.org/10.1111/j.1540-6261.2010.01625.x)
Almost three-fourths of the words a general dictionary counts as negative are not negative in financial text. The
paper is the reason financial NLP is a field of its own. Its remedy is a better word list, which is still a
function of words.

**Araci (2019); Yang, Uy and Huang (2020), "FinBERT."** [arXiv](https://arxiv.org/abs/1908.10063) · [arXiv](https://arxiv.org/abs/2006.08097)
Domain pretraining improves financial sentiment classification over general models. Both are evaluated on
sentence polarity, so they inherit the label's blind spots: Cioboata and Bildea (2026) invert the financial
meaning of sentences by targeted substitutions and deceive FinBERT in 90.85% of eligible cases.

**Sinha et al. (2022), "SEntFiN 1.0: Entity-Aware Sentiment Analysis for Financial News."** [doi](https://doi.org/10.1002/asi.24634)
10,753 headlines with entity-level sentiment, 2,847 with several entities and often conflicting sentiments. It is
the pool from which the Trial's headlines come. Its label is the sentiment of the news toward an entity; whether a
holder expresses a view on the entity as an investment is a further judgment, which is why every label was
reconsidered.

**Pei et al. (2022), "TweetFinSent."** [doi](https://doi.org/10.18653/v1/2022.finnlp-1.5)
Stock sentiment defined by whether the author gained or expects to gain a return, as distinct from the tone of
the tweet. It is the closest published definition to the stance axis, for one holder and one scope.

**Gyawali et al. (2025), "Evaluating Large Language Models for Stance Detection on Financial Targets."** [arXiv](https://arxiv.org/abs/2510.23464)
Sentence-level stance toward debt, earnings per share and sales in 10-K filings and earnings calls. It treats a
financial metric as a target, which is the metric-direction axis under another name, and evaluates prompting
strategies. The two axes are not yet put on the same text.

### Targets, holders and edits

**Liu (2012), *Sentiment Analysis and Opinion Mining*.** [doi](https://doi.org/10.2200/S00416ED1V01Y201204HLT016)
An opinion is a quintuple of entity, aspect, sentiment, holder and time. Equation (1.1) is that definition used as
the unit of supervision, with the evidence scope added and the sentiment split into task axes.

**Xing et al. (2020), "Tasty Burgers, Soggy Fries."** [arXiv](https://arxiv.org/abs/2009.07964)
Test sets that disentangle the target aspect from the others reduce the accuracy of nine aspect-sentiment models
by up to 69.73%. The models had learned the sentence's sentiment and not the target's. The paper builds a test;
the permission on a target swap is the corresponding training term.

**Ribeiro et al. (2020), "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList."** [arXiv](https://arxiv.org/abs/2005.04118)
Invariance tests are perturbations that should not change the output, and directional expectation tests are
perturbations with a known expected result. These are two of the four permissions, defined for testing a single
output. Permissions add the distinction inside invariance, between keeping the label and keeping the
distribution, and assign each per task axis.

**Gardner et al. (2020); Kaushik, Hovy and Lipton (2019).** [arXiv](https://arxiv.org/abs/2004.02709) · [arXiv](https://arxiv.org/abs/1909.12434)
Contrast sets and counterfactually revised documents: small, meaningful perturbations that change the gold label,
written by people. Classifiers trained on original data fail on the revised counterparts and the reverse. They
supply relabelling pairs and train on them with cross-entropy at each end. The relative-response term uses the
pairing itself.

**Abraham et al. (2022), "CEBaB."** [arXiv](https://arxiv.org/abs/2205.14140)
Restaurant reviews with human-written counterfactuals in which one aspect is modified, rated at aspect and review
level. The data structure, one edit with effects recorded on several levels, is the permission matrix of
Section 2 in another domain.

### Consistency and calibration

**Xie et al. (2019), "Unsupervised Data Augmentation for Consistency Training."** [arXiv](https://arxiv.org/abs/1904.12848)
Consistency training constrains predictions to be invariant to input noise, and better noise, such as
back-translation, gives better results. Back-translation is close to an equivalence, which is why it works.
Applied to an edit that adds evidence, the same loss pulls two different posteriors together, by the amount of
Equation (3.2).

**Liu et al. (2022), "Soft Augmentation for Image Classification."** [arXiv](https://arxiv.org/abs/2211.04625)
The learning target softens as the transform becomes more aggressive, and the resulting models are better
calibrated. It is the same observation from the data side: an augmentation can keep the label and change the
warranted confidence. The article states the cost of ignoring it on the loss side and gives a term that needs no
model of how much the confidence should change.

**Khazem (2026), "Margin and Consistency Supervision for Calibrated and Robust Vision Models."** [arXiv](https://arxiv.org/abs/2603.05812)
A margin penalty on the logit gap with a consistency term on mildly perturbed views improves calibration and
robustness in vision. The combination is the nearest in form. Its consistency acts on perturbations assumed to be
equivalences, and its margin is on the gap at one input, where the response term is on the change of the gap
across an edit.

**Guo et al. (2017), "On Calibration of Modern Neural Networks."** [arXiv](https://arxiv.org/abs/1706.04599)
Modern networks are poorly calibrated, and temperature scaling repairs most of it. The Trial uses it as the paper
intends, after selection and on separate data, and the fitted temperature of 1.506 is in the direction the paper
reports.

### Labels that belong to one reader

**Aroyo and Welty (2015); Plank (2022).** [doi](https://doi.org/10.1609/aimag.v36i1.2564) · [arXiv](https://arxiv.org/abs/2211.02570)
The single correct label is an idealization; variation among annotators is signal about the range of reasonable
readings. A model of one analyst's standard takes the argument to its end: the holder of the label is recorded,
and consistency is asked of one reader over time.

### Imbalance and what a score means

**King and Zeng (2001), "Logistic Regression in Rare Events Data."** [doi](https://doi.org/10.1093/oxfordjournals.pan.a004868)
Standard estimation underestimates the probability of rare events, and sampling designs can be corrected for.
The Trial's rebalancing is in this tradition: exposure is changed in training and the evaluation keeps the task's
shares.

**Menon et al. (2020), "Long-Tail Learning via Logit Adjustment."** [arXiv](https://arxiv.org/abs/2007.07314)
Adjusting logits by label frequencies, after training or in the loss, unifies several proposals for imbalance.
It is the alternative to resampling and the natural comparison for the next run.

**Opitz and Burst (2019), "Macro F1 and Macro F1."** [arXiv](https://arxiv.org/abs/1911.03347)
There are two formulas in use under one name, and they can differ by as much as 0.5. The article uses the mean of
the per-class F1 values throughout, and Equation (6.1) is stated for that definition.

### Training at scale

**Goyal et al. (2017), "Accurate, Large Minibatch SGD."** [arXiv](https://arxiv.org/abs/1706.02677)
Among its remarks on distributed training: normalize the per-worker loss by the total minibatch size, not the
per-worker size. Equation (4.2) is that remark generalized to several terms whose counts differ by orders of
magnitude and may be zero on a worker, with accumulation folded into the same window.

**Krell et al. (2021), "Efficient Sequence Packing without Cross-contamination."** [arXiv](https://arxiv.org/abs/2107.02027)
Up to half of all tokens in common datasets can be padding; packing recovers them if self-attention is kept from
mixing sequences. The paper works with encoders. For a causal decoder with rotary positions the article shows the
packed logits equal the unpacked ones with no change to positions, and measures it.

**Wijmans et al. (2024), "Cut Your Losses in Large-Vocabulary Language Models."** [arXiv](https://arxiv.org/abs/2411.09009)
The logits matrix dominates memory in training, and the cross-entropy can be computed without materializing it.
The method targets dense next-token loss. With sparse supervision a simpler step removes most of the matrix, and
the question that remains is which softmax is the objective.

**Su et al. (2021), "RoFormer."** [arXiv](https://arxiv.org/abs/2104.09864) **Ainslie et al. (2023), "GQA."** [arXiv](https://arxiv.org/abs/2305.13245) **Press and Wolf (2016).** [arXiv](https://arxiv.org/abs/1608.05859)
Rotary position embeddings encode position by rotation so that attention depends on relative position; grouped-query
attention uses fewer key-value heads than query heads; the input and output embeddings can be one matrix. The
architecture uses all three, and the packing result of Section 5 follows from the first.

**Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models."** [arXiv](https://arxiv.org/abs/2203.15556)
For compute-optimal training, model size and training tokens should be scaled equally. The recipe's 20.9
foundation tokens per parameter is set with that proportionality in mind.

### Models trained from scratch, and why

**Gu et al. (2020), "Domain-Specific Language Model Pretraining for Biomedical NLP."** [arXiv](https://arxiv.org/abs/2007.15779)
Where unlabelled domain text is abundant, pretraining from scratch outperforms continued pretraining of a general
model. It is the general argument for a domain vocabulary and a domain corpus.

**He et al. (2025a); Yan et al. (2026); Kelly et al. (2026).** [arXiv](https://arxiv.org/abs/2502.21206) · [arXiv](https://arxiv.org/abs/2603.11838) · [arXiv](https://arxiv.org/abs/2607.11889)
Chronologically consistent and point-in-time models: trained only on text available before a date, so that a
backtest cannot be contaminated by what the model has read. DatedGPT trains twelve 1.3B models on about 100
billion tokens each, and Kelly et al. reach 4B parameters on a trillion tokens. They establish that control over
the corpus requires training from scratch and that the cost in capability narrows with scale. The article uses the
same control for a different boundary, the texts that will be labelled.

**Wu et al. (2023), "BloombergGPT."** [arXiv](https://arxiv.org/abs/2303.17564)
A 50B-parameter model trained on 363 billion tokens of financial data and 345 billion of general text, evaluated
on financial and general benchmarks. It is the reference for what scale on proprietary financial text buys. Its
supervision is next-token prediction on that text; what the model should say about a holder's view of a target is
left to downstream use.

### Leakage

**Kaufman et al. (2012); Elangovan, He and Verspoor (2021); Søgaard et al. (2020).** [doi](https://doi.org/10.1145/2382577.2382579) · [arXiv](https://arxiv.org/abs/2102.01818) · [arXiv](https://arxiv.org/abs/2005.00636)
Leakage is information about the target that should not be available; overlap between training and test text
inflates results; random splits are optimistic. The family is the unit that answers all three for derived views,
and [D5](derivations.md#d5-families-views-and-siblings) gives the probability that a record-level split leaks.

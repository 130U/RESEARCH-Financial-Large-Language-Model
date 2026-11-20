# Results

Every figure in the article, recomputed.

179 of 179 figures agree with the article

## Label-preserving is not distribution-preserving (Section 3)

| Quantity | Recomputed | Article | Where | Agrees |
| --- | ---: | ---: | --- | :---: |
| Evidence an edit carries when a calibrated reader moves from 0.70 to 0.90 (nats) | 1.3499 | 1.35 | §3 | ✓ |
| The same as a likelihood ratio | 3.8571 | 3.86 | §3 | ✓ |
| Strength 0.03: prediction at the original | 0.7280 | 0.728 | §3 | ✓ |
| Strength 0.03: prediction at the edited text | 0.8720 | 0.872 | §3 | ✓ |
| Strength 0.03: calibration error at each end | 0.0280 | 0.028 | §3 | ✓ |
| Strength 0.03: the two errors are equal and opposite |  | holds | §3, eq. 3.2 | ✓ |
| Strength 0.1: prediction at the original | 0.7559 | 0.756 | §3 | ✓ |
| Strength 0.1: prediction at the edited text | 0.8441 | 0.844 | §3 | ✓ |
| Strength 0.1: calibration error at each end | 0.0559 | 0.056 | §3 | ✓ |
| Strength 0.1: the two errors are equal and opposite |  | holds | §3, eq. 3.2 | ✓ |
| Strength 0.3: prediction at the original | 0.7790 | 0.779 | §3 | ✓ |
| Strength 0.3: prediction at the edited text | 0.8210 | 0.821 | §3 | ✓ |
| Strength 0.3: calibration error at each end | 0.0790 | 0.079 | §3 | ✓ |
| Strength 0.3: the two errors are equal and opposite |  | holds | §3, eq. 3.2 | ✓ |
| Strength 1: prediction at the original | 0.7926 | 0.793 | §3 | ✓ |
| Strength 1: prediction at the edited text | 0.8074 | 0.807 | §3 | ✓ |
| Strength 1: calibration error at each end | 0.0926 | 0.093 | §3 | ✓ |
| Strength 1: the two errors are equal and opposite |  | holds | §3, eq. 3.2 | ✓ |
| Strength 3: prediction at the original | 0.7974 | 0.797 | §3 | ✓ |
| Strength 3: prediction at the edited text | 0.8026 | 0.803 | §3 | ✓ |
| Strength 3: calibration error at each end | 0.0974 | 0.097 | §3 | ✓ |
| Strength 3: the two errors are equal and opposite |  | holds | §3, eq. 3.2 | ✓ |
| Strength 10: prediction at the original | 0.7992 | 0.799 | §3 | ✓ |
| Strength 10: prediction at the edited text | 0.8008 | 0.801 | §3 | ✓ |
| Strength 10: calibration error at each end | 0.0992 | 0.099 | §3 | ✓ |
| Strength 10: the two errors are equal and opposite |  | holds | §3, eq. 3.2 | ✓ |
| Limit of the error as the strength grows | 0.1000 | 0.1 | §3, eq. 3.2 | ✓ |
| Both predictions in the limit | 0.8000 | 0.8 | §3 | ✓ |
| A true equivalence costs nothing at any strength |  | holds | §3 | ✓ |
| The error increases with the strength and stays below half the gap |  | holds | §3, eq. 3.2 | ✓ |
| Minimizing the implemented loss over free logits reaches the same predictions (strength 0.3) |  | holds | §3 | ✓ |
| Minimizing the implemented loss over free logits reaches the same predictions (strength 1) |  | holds | §3 | ✓ |
| Minimizing the implemented loss over free logits reaches the same predictions (strength 3) |  | holds | §3 | ✓ |
| Three classes: the error vectors at the two ends are equal and opposite |  | holds | §3 | ✓ |
| Three classes: at high strength both predictions approach the average posterior |  | holds | §3 | ✓ |
| Safe margin when labels are assigned at posterior odds of at least 2 (nats) | 1.3863 | 1.39 | §3, eq. 3.3 | ✓ |
| A calibrated model at odds of exactly 2 at both ends meets that margin with zero penalty |  | holds | §3, eq. 3.3 | ✓ |
| A model that does not respond pays the full margin |  | holds | §3 | ✓ |

## One update, one objective (Section 4)

| Quantity | Recomputed | Article | Where | Agrees |
| --- | ---: | ---: | --- | :---: |
| Training examples in the task pool | 705 | 705 | §4 | ✓ |
| Task pool | 1,008 | 1,008 | §4 | ✓ |
| Families of 2 views: chance that a held-out view has a sibling in training (%) | 69.9405 | 69.9 | §4 | ✓ |
| Families of 3 views: chance that a held-out view has a sibling in training (%) | 90.9643 | 91 | §4 | ✓ |
| Families of 5 views: chance that a held-out view has a sibling in training (%) | 99.1836 | 99.2 | §4 | ✓ |
| A family with twenty edits weighs as much as a family with none |  | holds | §4 | ✓ |
| Stance and event inputs as a share of adaptation tokens (%) | 1 | 1 | §4 | ✓ |
| Cells of 2 sequences that contain a term carried by 1% of sequences (%) | 1.9900 | 1.99 | §4 | ✓ |
| Cells of 8 sequences that contain a term carried by 1% of sequences (%) | 7.7255 | 7.7 | §4 | ✓ |
| Cells of 32 sequences that contain a term carried by 1% of sequences (%) | 27.5020 | 27.5 | §4 | ✓ |
| Sequences in the window | 16 | 16 | §4 | ✓ |
| Language positions | 756 | 756 | §4 | ✓ |
| Answer positions | 16 | 16 | §4 | ✓ |
| Stance positions | 2 | 2 | §4 | ✓ |
| Ranks | 4 | 4 | §4 | ✓ |
| Microbatches per rank | 2 | 2 | §4 | ✓ |
| Global denominators: gradient equal to the single-batch gradient up to single-precision rounding (measured 9.8e-07) |  | about 1e-6 | §4, eq. 4.2 | ✓ |
| Per-cell averaging: relative difference from the single-batch gradient | 0.8588 | 0.86 | §4 | ✓ |
| Cells that contain the stance term, of eight | 1 | 1 | §4 | ✓ |
| Cells that contain the answer term, of eight | 2 | 2 | §4 | ✓ |

## Execution that preserves the objective (Section 5)

| Quantity | Recomputed | Article | Where | Agrees |
| --- | ---: | ---: | --- | :---: |
| Selected-position loss equals the all-position loss |  | holds | §5, eq. 5.1 | ✓ |
| Their gradients agree up to single-precision rounding (measured 6.4e-07) |  | about 1e-6 | §5, eq. 5.1 | ✓ |
| 2.4B microbatch, all positions: logits tensor (GiB) | 1 | 1 | §5 | ✓ |
| Its two supervised positions (MiB) | 0.5000 | 0.5 | §5 | ✓ |
| Ratio | 2,048 | 2,048 | §5 | ✓ |
| Trial batch of 4 × 512, all positions (MiB) | 64 | 64 | §5 | ✓ |
| Its four supervised positions (KiB) | 128 | 128 | §5 | ✓ |
| Output head's share of a position's matrix work at 2.4B (%) | 6.9984 | 7 | §5 | ✓ |
| At 24M (%) | 17.2973 | 17.3 | §5 | ✓ |
| Full-vocabulary loss minus label-restricted loss, two positions (nats) | 15.6640 | 15.66 | §5 | ✓ |
| The same per position (nats) | 7.8320 | 7.8 | D6 | ✓ |
| Log of the vocabulary size over three label tokens (nats) | 7.9123 | 7.9 | D6 | ✓ |
| Ratio of a term's kept weight between cells of 32 and of 2 sequences | 13.8201 | 14 | D4 | ✓ |
| That difference is minus the log of the probability mass on the label tokens |  | holds | §5 | ✓ |
| Examples packed | 3 | 3 | §5 | ✓ |
| Packed length | 64 | 64 | §5 | ✓ |
| Segmented attention: packed logits within 1e-5 of the unpacked ones (measured 2.2e-06) |  | holds | §5, eq. 5.2 | ✓ |
| Plain causal attention: largest logit difference | 2.9074 | 2.9 | §5 | ✓ |
| Typical logit magnitude | 0.3598 | 0.36 | §5 | ✓ |
| Segmented attention: gradient into an earlier segment | 0 | 0 | §5 | ✓ |
| Plain causal attention: gradient into an earlier segment | 0.8764 | 0.88 | §5 | ✓ |
| The input embedding and the output head are one parameter object |  | holds | §5 | ✓ |
| 2.4B training state in single precision (GB) | 38.3594 | 38.36 | §5 | ✓ |

## The Trial (Section 6)

| Quantity | Recomputed | Article | Where | Agrees |
| --- | ---: | ---: | --- | :---: |
| Trial parameters | 24,255,744 | 24,255,744 | §6 | ✓ |
| The formula gives the same count |  | holds | §6, eq. 7.1 | ✓ |
| Tokenizer vocabulary | 8,192 | 8,192 | §6 | ✓ |
| Task texts excluded from the tokenizer and the language corpora | 10,686 | 10,686 | §6 | ✓ |
| Throughput between 3,200 and 3,500 tokens per second in every stage |  | holds | §6 | ✓ |
| Peak allocated memory (GiB) | 1.0628 | 1.06 | §6 | ✓ |
| Sequences per microbatch | 4 | 4 | §6 | ✓ |
| Microbatches per update | 8 | 8 | §6 | ✓ |
| Seed: updates | 300 | 300 | §6 | ✓ |
| Seed: supervised tokens | 4,914,941 | 4,914,941 | §6 | ✓ |
| Seed: validation loss | 5.5419 | 5.542 | §6 | ✓ |
| Expansion: updates | 1,220 | 1,220 | §6 | ✓ |
| Expansion: supervised tokens | 19,988,195 | 19,988,195 | §6 | ✓ |
| Expansion: validation loss | 3.6929 | 3.693 | §6 | ✓ |
| Expansion: loss on the seed's validation text before | 4.4032 | 4.403 | §6 | ✓ |
| Expansion: after | 3.9842 | 3.984 | §6 | ✓ |
| New general training tokens available to the expansion (millions) | 41.4994 | 41.5 | evidence | ✓ |
| Adaptation: updates | 74 | 74 | §6 | ✓ |
| Adaptation: supervised tokens | 1,211,960 | 1,211,960 | §6 | ✓ |
| Adaptation: validation loss on financial text | 3.2375 | 3.238 | §6 | ✓ |
| Adaptation: loss on general text before | 3.9842 | 3.984 | §6 | ✓ |
| Adaptation: after | 4.0574 | 4.057 | §6 | ✓ |
| Cost of adaptation on general text (nats) | 0.0732 | 0.073 | §6 | ✓ |
| Financial share of the adaptation mixture (%) | 80 | 80 | §6 | ✓ |
| Supervised tokens in all three stages | 26,115,096 | 26,115,096 | §6 | ✓ |
| Supervised tokens per parameter | 1.0767 | 1.08 | §6 | ✓ |
| Headlines reviewed | 2,000 | 2,000 | §6 | ✓ |
| Decisions confirmed | 1,993 | 1,993 | §6 | ✓ |
| Of which exclusions | 114 | 114 | §6 | ✓ |
| Unresolved | 7 | 7 | §6 | ✓ |
| Three-class examples | 1,879 | 1,879 | §6 | ✓ |
| Confirmed decisions less exclusions give the three-class examples, and the three classes add up |  | holds | §6 | ✓ |
| Neutral | 1,219 | 1,219 | §6 | ✓ |
| Bullish | 426 | 426 | §6 | ✓ |
| Bearish | 234 | 234 | §6 | ✓ |
| Development, calibration and test examples each | 101 | 101 | §6 | ✓ |
| The four partitions add up to the task pool |  | holds | §6 | ✓ |
| Training: neutral | 491 | 491 | §6 | ✓ |
| Training: bullish | 140 | 140 | §6 | ✓ |
| Training: bearish | 74 | 74 | §6 | ✓ |
| First model: epochs | 2 | 2 | §6 | ✓ |
| First model: updates | 45 | 45 | §6 | ✓ |
| It predicted neutral for every development and test example |  | holds | §6 | ✓ |
| Development: accuracy of the majority identity (%) | 75.2475 | 75.25 | §6, eq. 6.1 | ✓ |
| Development: macro-F1 of the majority identity | 0.2863 | 0.2863 | §6, eq. 6.1 | ✓ |
| The first model's scores equal the identity exactly |  | holds | §6 | ✓ |
| Test: accuracy of the identity (%) | 58.4158 | 58.42 | §6, eq. 6.1 | ✓ |
| Test: macro-F1 of the identity | 0.2458 | 0.2458 | §6, eq. 6.1 | ✓ |
| Cross-entropy of a predictor that outputs the training shares (nats) | 0.8096 | 0.81 | §6 | ✓ |
| Rebalanced run: exposures per class per epoch | 491 | 491 | §6 | ✓ |
| Epochs | 3 | 3 | §6 | ✓ |
| Updates | 139 | 139 | §6 | ✓ |
| Label positions | 4,419 | 4,419 | §6 | ✓ |
| Label positions equal classes × exposures × epochs |  | holds | §6 | ✓ |
| Bearish: support | 10 | 10 | §6 | ✓ |
| Bearish: precision | 0.4545 | 0.4545 | §6 | ✓ |
| Bearish: recall | 0.5000 | 0.5 | §6 | ✓ |
| Bearish: F1 | 0.4762 | 0.4762 | §6 | ✓ |
| Neutral: support | 76 | 76 | §6 | ✓ |
| Neutral: precision | 0.9394 | 0.9394 | §6 | ✓ |
| Neutral: recall | 0.8158 | 0.8158 | §6 | ✓ |
| Neutral: F1 | 0.8732 | 0.8732 | §6 | ✓ |
| Bullish: support | 15 | 15 | §6 | ✓ |
| Bullish: precision | 0.5000 | 0.5 | §6 | ✓ |
| Bullish: recall | 0.8000 | 0.8 | §6 | ✓ |
| Bullish: F1 | 0.6154 | 0.6154 | §6 | ✓ |
| Rebalanced model: macro-F1 | 0.6549 | 0.6549 | §6 | ✓ |
| Rebalanced model: accuracy (%) | 78.2178 | 78.22 | §6 | ✓ |
| TF-IDF baseline: macro-F1 | 0.6087 | 0.6087 | §6 | ✓ |
| TF-IDF baseline: accuracy (%) | 75.2500 | 75.25 | §6 | ✓ |
| Directional examples gained | 17 | 17 | §6 | ✓ |
| Neutral examples given up | 14 | 14 | §6 | ✓ |
| Correct after | 79 | 79 | §6 | ✓ |
| Correct before | 76 | 76 | §6 | ✓ |
| Gain in macro-F1 | 0.3687 | 0.3687 | §6 | ✓ |
| Margin over the TF-IDF baseline | 0.0462 | 0.0462 | §6 | ✓ |
| Fitted temperature | 1.5063 | 1.506 | §6 | ✓ |
| A raw confidence of 0.90 after temperature scaling | 0.8113 | 0.81 | §6 | ✓ |
| Review threshold | 0.5738 | 0.574 | §6 | ✓ |
| Calibration examples for the temperature | 51 | 51 | §6 | ✓ |
| For the threshold | 50 | 50 | §6 | ✓ |

## Two scales, one contract (Section 7)

| Quantity | Recomputed | Article | Where | Agrees |
| --- | ---: | ---: | --- | :---: |
| 2.4B parameters | 2,397,459,968 | 2,397,459,968 | §7, eq. 7.1 | ✓ |
| Parameters per layer at 2.4B | 79,631,616 | 79,631,616 | §7 | ✓ |
| Parameters per layer at 24M | 3,343,488 | 3,343,488 | §7 | ✓ |
| Twenty-eight layers | 2,229,685,248 | 2,229,685,248 | §7 | ✓ |
| Shared vocabulary matrix | 167,772,160 | 167,772,160 | §7 | ✓ |
| Final norm | 2,560 | 2,560 | §7 | ✓ |
| Head dimension at 2.4B | 128 | 128 | §7 | ✓ |
| Head dimension at 24M | 64 | 64 | §7 | ✓ |
| Foundation tokens (billions) | 50 | 50 | §7 | ✓ |
| Adaptation tokens (billions) | 5 | 5 | §7 | ✓ |
| Financial source text (billions) | 4 | 4 | §7 | ✓ |
| General text (billions) | 0.7500 | 0.75 | §7 | ✓ |
| Reading questions and answers (billions) | 0.2000 | 0.2 | §7 | ✓ |
| Stance and event tasks (billions) | 0.0500 | 0.05 | §7 | ✓ |
| Foundation tokens per parameter | 20.8554 | 20.9 | §7 | ✓ |
| Human-labelled records | 32,000 | 32,000 | §7 | ✓ |
| Scale factor between the two configurations | 98.8409 | 100 | §7 | ✓ |

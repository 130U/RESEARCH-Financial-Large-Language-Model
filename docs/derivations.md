# Derivations

Proofs of the article's results, in its notation and with its equation numbers. Section numbers (§) refer to the
[article](../manuscript/semantic-permissions-and-objective-preserving-scaling.md); every figure quoted here is
recomputed in [results](../reproduce/results.md).

| | Result | Section |
| --- | --- | --- |
| [D1](#d1-the-price-of-unlicensed-consistency) | The price of unlicensed consistency | §3, eq. 3.2 |
| [D2](#d2-the-relative-response-constraint-is-free-for-a-calibrated-model) | The relative-response constraint is free for a calibrated model | §3, eq. 3.3 |
| [D3](#d3-one-update-one-objective) | One update, one objective | §4, eqs. 4.1 and 4.2 |
| [D4](#d4-what-per-cell-averaging-does) | What per-cell averaging does | §4 |
| [D5](#d5-families-views-and-siblings) | Families, views and siblings | §4 |
| [D6](#d6-projecting-only-supervised-positions) | Projecting only supervised positions | §5, eq. 5.1 |
| [D7](#d7-segmented-attention-on-a-rotary-model) | Segmented attention on a rotary model | §5, eq. 5.2 |
| [D8](#d8-a-constant-predictor-has-an-exact-signature) | A constant predictor has an exact signature | §6, eq. 6.1 |
| [D9](#d9-temperature-cannot-repair-a-collapse) | Temperature cannot repair a collapse | §6 |
| [D10](#d10-counting-parameters-and-state) | Counting parameters and state | §7, eq. 7.1 |

---

## D1. The price of unlicensed consistency

**Two classes.** Let $`p, p'`$ be the true posteriors of one class at the original and the edited text, and
$`s, s'`$ the model's log-odds, with $`q = \sigma(s)`$ and $`q' = \sigma(s')`$. The expected loss on the pair with a
consistency term of strength $`\lambda`$ is

```math
J(s, s') = \mathrm{CE}(p; q) + \mathrm{CE}(p'; q') + \frac{\lambda}{2}(s - s')^2 ,
\qquad \mathrm{CE}(p; q) = -p\log\,q - (1-p)\log(1-q) . \qquad (D1.1)
```

$`J`$ is strictly convex in $`(s, s')`$, since each cross-entropy is strictly convex in its log-odds and the
penalty is convex. Its stationary point is its minimum. With $`\partial\,\mathrm{CE}(p; \sigma(s))/\partial s =
\sigma(s) - p`$,

```math
q - p + \lambda(s - s') = 0, \qquad q' - p' - \lambda(s - s') = 0 . \qquad (D1.2)
```

Adding the two gives $`q + q' = p + p'`$: the prediction errors at the two ends are equal and opposite. Writing
$`\varepsilon = q - p = p' - q'`$,

```math
\varepsilon \;=\; \lambda\,(s' - s) \;=\; \lambda\,\big[\mathrm{logit}(p' - \varepsilon) - \mathrm{logit}(p + \varepsilon)\big] . \qquad (D1.3)
```

Take $`p < p'`$. The right side is continuous and strictly decreasing in $`\varepsilon`$ on
$`[0, (p'-p)/2]`$, equal to $`\lambda(\mathrm{logit}\,p' - \mathrm{logit}\,p) > 0`$ at $`\varepsilon = 0`$
and to $`0`$ at $`\varepsilon = (p'-p)/2`$, where the two arguments coincide. The left side increases from $`0`$.
So there is exactly one root, and it lies strictly inside the interval: $`0 < \varepsilon(\lambda) < (p'-p)/2`$.
Increasing $`\lambda`$ raises the right side at every $`\varepsilon`$ and therefore moves the root to the right, so
$`\varepsilon`$ is increasing in $`\lambda`$. As $`\lambda \to \infty`$ the root must satisfy
$`\mathrm{logit}(p' - \varepsilon) - \mathrm{logit}(p + \varepsilon) \to 0`$, that is
$`\varepsilon \to (p'-p)/2`$. If $`p = p'`$ the root is $`\varepsilon = 0`$ for every $`\lambda`$. ∎ (Equation 3.2)

**$`K`$ classes.** With posteriors $`p, p' \in \Delta^{K-1}`$, logits $`z, z'`$, $`q = \mathrm{softmax}(z)`$,
and the consistency term $`\tfrac{\lambda}{2}\lVert \bar z - \bar z'\rVert^2`$ on mean-removed logits, the
stationarity conditions are

```math
q - p + \lambda(\bar z - \bar z') = 0, \qquad q' - p' - \lambda(\bar z - \bar z') = 0 , \qquad (D1.4)
```

because the gradient of the penalty with respect to $`z`$ is $`\lambda(\bar z - \bar z')`$, the centering matrix
being a projection. Hence $`q - p = -(q' - p')`$: the error vectors at the two ends are equal and opposite, class
by class. As $`\lambda \to \infty`$ the minimizer has $`\bar z = \bar z'`$, so $`q = q'`$, and the common value
minimizes $`\mathrm{CE}(p; q) + \mathrm{CE}(p'; q)`$, which is $`q = (p + p')/2`$. Each end is then in error by
half the difference of the posteriors, a total-variation error of $`\tfrac12\,\mathrm{TV}(p, p')`$. For two
classes $`\lVert \bar z - \bar z'\rVert^2 = \tfrac12 (s - s')^2`$, which is why the implemented term is the squared
distance without the factor one half.

**Any consistency penalty has the same limit.** If a penalty $`\lambda\, d(q, q')`$ has $`d = 0`$ exactly when
$`q = q'`$, the limit $`\lambda \to \infty`$ forces $`q = q'`$ and the anchors then choose $`(p + p')/2`$. The
limit error does not depend on whether $`d`$ is a squared distance of log-odds, a Kullback–Leibler divergence in
either direction or a Jensen–Shannon divergence; only the path at finite $`\lambda`$ does.

**Worked case.** $`p = 0.70`$, $`p' = 0.90`$. The edit's evidence is
$`\mathrm{logit}\,0.90 - \mathrm{logit}\,0.70 = \log(27/7) = 1.35`$ nats, a likelihood ratio of 3.86. At
$`\lambda = 1`$ the root is $`\varepsilon = 0.093`$, giving predictions 0.793 and 0.807; the limit is 0.100. The
code solves the root by bisection and, separately, minimizes the implemented pair loss over free logits; the two
agree to six digits.

## D2. The relative-response constraint is free for a calibrated model

For a relabelling edit from $`a`$ to $`b`$ the constraint is $`\max(0, m - D)`$ with
$`D = (z'_b - z'_a) - (z_b - z_a)`$. A calibrated model has $`z_b - z_a = \log(p_b/p_a)`$ at each end, so

```math
D^* \;=\; \log\,\frac{p'_b}{p'_a} - \log\,\frac{p_b}{p_a} . \qquad (D2.1)
```

This is the evidence the edit carries for $`b`$ against $`a`$. The constraint is zero at the calibrated predictor
exactly when $`m \le D^*`$. The anchors are minimized by the calibrated predictor and the constraint adds nothing
there, so for $`m \le D^*`$ the calibrated predictor minimizes the whole pair loss: the constraint costs no
calibration.

**A margin that is safe for a whole edit set.** Suppose a label is assigned only when its posterior odds against
the competing label are at least $`r`$. Then $`p_a/p_b \ge r`$ at the original and $`p'_b/p'_a \ge r`$ at the
edited text, and

```math
D^* \;=\; \log\,\frac{p'_b}{p'_a} + \log\,\frac{p_a}{p_b} \;\ge\; 2\log\,r . \qquad (D2.2)
```

Every margin $`m \le 2\log\,r`$ is therefore safe on every pair. For $`r = 2`$, $`m \le 1.39`$ nats. ∎ (Equation 3.3)

**What it still does.** A model whose output does not move across the edit has $`D = 0`$ and pays $`m`$. A model
that moves the wrong way pays more. The constraint is one-sided: it asks for a response of at least $`m`$ in the
known direction and leaves the magnitude above $`m`$ to the anchors. That is the property a symmetric penalty
lacks, and it is why label-preserving pairs, where the direction of the change in confidence is known and its size
is not, receive no symmetric term.

## D3. One update, one objective

Let the window be the multiset of examples consumed between two optimizer steps, and let term $`j`$ have per-position
losses $`\ell_{j,i}`$ for $`i`$ in the window, with $`S_j = \sum_i \ell_{j,i}`$ and effective count $`C_j`$. The
objective is $`L = \sum_{j: C_j > 0} \lambda_j S_j / C_j`$ (Equation 4.1).

Partition the window into cells $`(r, k)`$, $`r = 1..W`$ ranks and $`k = 1..A_r`$ microbatches, with
$`S_{rk,j}`$ the sum of term $`j`$ in the cell, so that $`\sum_{r,k} S_{rk,j} = S_j`$. Let the cell's loss be
$`\ell_{rk} = W \sum_{j: C_j>0} \lambda_j S_{rk,j}/C_j`$. Gradients are summed over the microbatches of a rank and
averaged over ranks:

```math
\frac{1}{W}\sum_{r=1}^{W}\sum_{k=1}^{A_r} \nabla \ell_{rk}
\;=\; \sum_{j:\,C_j>0} \frac{\lambda_j}{C_j} \sum_{r,k} \nabla S_{rk,j}
\;=\; \sum_{j:\,C_j>0} \frac{\lambda_j}{C_j}\, \nabla S_j \;=\; \nabla L . \quad ∎ \qquad (D3.1)
```

Nothing was assumed about the partition: ranks may hold different numbers of microbatches, cells different
numbers of examples, and a term may be absent from any cell. A term absent from the whole window has $`C_j = 0`$
and is left out of the sum; a rank on which it is absent still takes part in the same reduction. The identity is
Equation (4.2). With family-aware terms the per-view weights of [D5](#d5-families-views-and-siblings) are folded
into $`\ell_{j,i}`$ and $`C_j`$, and the argument is unchanged.

**Measured.** On the Trial model, with 756 language positions, 16 answer positions and 2 stance positions in a
window of 16 sequences, the gradient from four ranks of two microbatches differs from the single-batch gradient by
a relative $`10^{-6}`$ in single precision.

## D4. What per-cell averaging does

Let each cell average term $`j`$ over its own count $`n_{rk,j}`$ and let the cells be averaged. An example of term
$`j`$ in cell $`(r,k)`$ then has weight

```math
\frac{\lambda_j}{N\, n_{rk,j}} \qquad \text{in place of} \qquad \frac{\lambda_j}{C_j},
\qquad N = \text{number of cells} . \qquad (D4.1)
```

The ratio of the two is $`\bar n_j / n_{rk,j}`$ with $`\bar n_j = C_j / N`$: an example is over-weighted in a cell
that holds few of its kind and under-weighted in a crowded one. Summing the weights of term $`j`$ over the window
gives

```math
\sum_{(r,k):\, n_{rk,j} > 0} n_{rk,j}\cdot\frac{\lambda_j}{N\,n_{rk,j}} \;=\; \lambda_j\,\frac{\#\{\text{cells that contain } j\}}{N} \;=\; \lambda_j\,\pi_j . \qquad (D4.2)
```

The term keeps the share $`\pi_j`$ of its weight. A dense term has $`\pi_j = 1`$. A sparse one does not, and its
$`\pi_j`$ depends on the cell size, that is, on the batch size and the number of devices. If a share $`q`$ of
sequences carries the term and a cell holds $`n`$ independently drawn sequences, $`\pi = 1 - (1-q)^n`$: for
$`q = 1\%`$, 1.99% at $`n = 2`$, 7.7% at $`n = 8`$ and 27.5% at $`n = 32`$. The same recipe then trains the term at
weights that differ by a factor of about fourteen between a small cell and a large one.

In the measured window the stance term is in one cell of eight and the answer term in two, so they keep
$`1/8`$ and $`1/4`$ of their weights, and the gradient differs from the intended one by 0.86 in relative norm.

## D5. Families, views and siblings

**Weights.** Within a term, views are averaged inside a family and families are averaged. A view of family $`f`$
has weight $`1/(F\,\left\vert V(f)\right\vert)`$, where $`F`$ is the number of families and $`\left\vert V(f)\right\vert`$ the number of the family's
views in the term. The weights sum to one, and every family's views sum to $`1/F`$: a family with twenty edits and
a family with one view contribute equally.

**Siblings.** Let a family have $`k`$ views and let views be assigned to training independently with probability
$`\rho`$. A held-out view has at least one sibling in training with probability $`1 - (1-\rho)^{k-1}`$. At the
Trial's share $`\rho = 705/1008`$ this is 69.9% for $`k = 2`$, 91.0% for $`k = 3`$ and 99.2% for $`k = 5`$.
Assigning the family as a unit makes it zero. This is the reason families are split before views are generated,
and the reason the task pool's exclusion from the tokenizer and the language corpora is applied to source texts
before any derived text exists.

## D6. Projecting only supervised positions

Let $`H \in ℝ^{B\times S\times d}`$ be the final hidden states, $`ℳ`$ the set of supervised
positions, $`M = \left\vert ℳ\right\vert`$, and $`W_{\text{vocab}} \in ℝ^{V\times d}`$ the vocabulary matrix.
The loss is $`\sum_{(b,s)\in ℳ} \mathrm{CE}(W_{\text{vocab}} H_{b,s},\, y_{b,s})`$. It depends on $`H`$ only
through the rows in $`ℳ`$. Computing $`W_{\text{vocab}} H_{b,s}`$ for every $`(b,s)`$ and discarding the rest
produces the same numbers for the rows that are kept, so the loss is identical, and so are its gradients with
respect to $`W_{\text{vocab}}`$ and to $`H`$, the gradient at an unsupervised row being zero in both
computations. ∎ (Equation 5.1)

**Size.** The logits tensor has $`B\,S\,V`$ entries in one case and $`M\,V`$ in the other. In single precision,
$`B = 2`$, $`S = 2{,}048`$, $`V = 65{,}536`$ give $`2^{1+11+16+2} = 2^{30}`$ bytes, 1 GiB; $`M = 2`$ gives
$`2^{19}`$ bytes, 0.5 MiB; the ratio is $`BS/M = 2{,}048`$. At $`B = 4`$, $`S = 512`$, $`V = 8{,}192`$: 64 MiB and,
for $`M = 4`$, 128 KiB.

**Work.** A position's forward matrix work is proportional to the weights it is multiplied by. The head's share
is $`dV / (L\,(2d^2 + 2dgh + 3df) + dV)`$: 7.0% at 2.4B and 17.3% at 24M.

**Restricting the softmax is a different loss.** Let $`𝒴`$ be the label tokens and
$`\mu = \sum_{y\in 𝒴} \mathrm{softmax}(z)_y`$ the mass on them. For a target
$`y^* \in 𝒴`$,

```math
\mathrm{CE}_{\text{full}}(z, y^*) \;=\; -\log\,\frac{e^{z_{y^*}}}{\sum_{v} e^{z_v}}
\;=\; -\log\,\frac{e^{z_{y^*}}}{\sum_{y\in 𝒴} e^{z_y}} \;-\; \log\,\mu
\;=\; \mathrm{CE}_{𝒴}(z, y^*) - \log\,\mu . \qquad (D6.1)
```

The full-vocabulary loss exceeds the restricted one by $`-\log\,\mu \ge 0`$, and the gradient of that term is what
moves mass from the rest of the vocabulary onto the labels. On the Trial model at initialization, with three label
tokens, the difference is 15.66 nats over two positions, about 7.8 nats each, close to $`\log(8192/3) = 7.9`$.

## D7. Segmented attention on a rotary model

**No information crosses.** With $`\text{allowed}(i,j) = 𝟏[\mathrm{seg}(i) = \mathrm{seg}(j)]\,
𝟏[j \le i]`$ (Equation 5.2), the attention output at position $`i`$ in any layer is a function of the
layer's inputs at positions of the same segment not after $`i`$. By induction over layers, the hidden state at
$`i`$ is a function of the tokens of its own segment up to $`i`$ only. Its derivative with respect to any token of
another segment is identically zero. ∎

**The values are those of the example alone.** Rotary embeddings rotate the query at position $`i`$ by an angle
$`i\theta`$ and the key at $`j`$ by $`j\theta`$ in each two-dimensional subspace, so their inner product depends
on $`(i - j)\theta`$. A segment that starts at offset $`o`$ has positions $`o, o+1, \dots`$, and every pair inside
it has the same difference as in the unpacked example. All attention scores inside the segment are therefore the
same as when the example stands alone, the feed-forward and normalization layers act position by position, and
the logits are equal. No position index is reset. A model with absolute position embeddings would need segment-local
position ids to obtain the same equality.

**Measured.** Three examples of 20, 24 and 20 tokens packed into 64 positions: the largest difference between
packed and unpacked logits is within $`10^{-5}`$ with the mask and 2.9 without it, against a typical logit
magnitude of 0.36. The gradient of the second segment's outputs with respect to the first segment's input
embeddings is exactly 0 with the mask and 0.88 at its largest entry without it.

## D8. A constant predictor has an exact signature

Let a predictor answer the same class for every example, and let that class have share $`\pi`$ among $`n`$
examples of $`K`$ classes. Its accuracy is $`\pi`$. On the predicted class, precision is $`\pi`$ and recall is 1,
so $`F_1 = 2\pi/(1+\pi)`$. On every other class recall is 0 and $`F_1 = 0`$. Hence

```math
\text{macro-}F_1 \;=\; \frac{1}{K}\cdot\frac{2\pi}{1+\pi} . \quad ∎ \qquad (D8.1)
```

This is Equation (6.1). Development: $`\pi = 76/101`$ gives accuracy 75.25% and
macro-$`F_1 = \tfrac13\cdot\tfrac{152}{177} = 0.2863`$. Test: $`\pi = 59/101`$ gives 58.42% and 0.2458. Any reported
pair of accuracy and macro-$`F_1`$ can be tested against this curve before it is interpreted; a point on the curve
is a constant predictor.

**Why the first run sat on the curve.** A predictor that outputs the training shares $`(491, 140, 74)/705`$
has cross-entropy equal to their entropy, 0.81 nats. The label logits' bias direction reaches that value quickly,
and the loss then falls only as fast as features are learned. With 45 updates the run ended in that regime, where
the argmax is the majority class for every input.

## D9. Temperature cannot repair a collapse

Temperature scaling replaces logits $`z`$ by $`z/T`$ with $`T > 0`$. Division by a positive number preserves the
order of the entries, so $`\arg\max_k z_k/T = \arg\max_k z_k`$: the predicted class is the same at every
temperature. A model that predicts one class for every input does so after calibration too. ∎ Calibration and
class balance are therefore handled by different steps: the second by exposure during training, the first by a
temperature fitted afterwards on examples used for nothing else.

**What the fitted temperature means.** For two classes, dividing the log-odds by $`T`$ maps a confidence $`c`$ to
$`\sigma(\mathrm{logit}(c)/T)`$. At $`T = 1.506`$, $`c = 0.90`$ becomes 0.81. A temperature above 1 means the
raw model was overconfident.

## D10. Counting parameters and state

A layer has a query projection $`d \times d`$, key and value projections $`d \times gh`$ each, for $`g`$
key-value heads of dimension $`h`$, an output projection $`d \times d`$, three feed-forward matrices $`d\times f`$
(gate, up and down of a SwiGLU block), two RMSNorm scale vectors of length $`d`$, and two QK-norm scale vectors of
length $`h`$. The embedding and the vocabulary head share one $`V \times d`$ matrix, and there is one final norm:

```math
P \;=\; L\,\big(2d^2 + 2dgh + 3df + 2d + 2h\big) + Vd + d . \qquad (D10.1)
```

This is Equation (7.1).

| | $`L`$ | $`d`$ | $`f`$ | $`g`$ | $`h`$ | $`V`$ | Per layer | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24M | 6 | 512 | 1,664 | 4 | 64 | 8,192 | 3,343,488 | 24,255,744 |
| 2.4B | 28 | 2,560 | 8,320 | 4 | 128 | 65,536 | 79,631,616 | 2,397,459,968 |

The Trial's count equals the number of parameters of the instantiated model, which the tests assert.

**State.** Weights, gradients and the two Adam moments are four tensors of $`P`$ values. In single precision the
2.4B model's state is $`16P = 38{,}359{,}359{,}488`$ bytes, 38.36 GB, before activations.

**Tokens per parameter.** 50B foundation tokens over the 2.4B model's $`P`$ is 20.9. The Trial's 26,115,096
supervised tokens are 1.08 per parameter.

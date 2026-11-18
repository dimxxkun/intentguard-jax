# Experiment protocol

## Source and audit

The source is CLINC150 `data_full.json`, pinned to upstream commit `828f8093932c8fe6ca7936c3d2e52903b1c523de` with SHA-256 `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`. `prepare` checks these bytes before parsing. See the [dataset attribution](../THIRD_PARTY_NOTICES.md).

The audit canonicalizes text with Unicode NFKC, case folding, and whitespace collapse. If a canonical text occurs more than once across the source partitions, **every occurrence is quarantined**, regardless of its labels. Five duplicate groups were found, four with conflicting labels, removing ten records: five train, three validation, and two test examples. IDs and source locations are retained in [audit.json](../results/audit.json).

This is a text-identity audit across all partitions before model fitting. It does inspect the supplied split contents and reports label conflicts; it does not use model predictions or test outcomes to choose exclusions. It does not detect semantic paraphrases or establish that all possible leakage is absent. Cleaning changes the benchmark, so these results should not be compared directly with published scores on the untouched splits.

## Separate ownership of development decisions

| Role | In-scope count | OOS count | Permitted use |
|---|---:|---:|---|
| Train | 14,995 | 0 | Tokenizer and model/baseline fitting |
| Selection | 1,200 | 0 | Minimum-NLL checkpoint, early stopping |
| Temperature | 600 | 0 | One scalar temperature |
| Conformal | 600 | 0 | Prediction-set score quantile |
| Policy | 597 | 100 | Routing threshold under empirical OOS budget |
| Test | 4,498 | 1,000 | Frozen-model evaluation |

The original 100 OOS training examples are deliberately unused. Within each in-scope validation class, deterministic ordering and a seeded shuffle precede five approximately equal partitions. The first two are combined for selection; the remaining three have separate calibration roles. OOS validation examples are assigned only to policy fitting. The generated manifest hashes each role file and its label order; readers verify those hashes.

The model-fitting function reads only train and selection. Calibration reads only temperature, conformal, and policy. A test physically removes the test artifact before fitting the decision rule to verify this boundary. The baseline learns its vocabulary and coefficients only on train; its fixed hyperparameters are not selected on test.

## Training and model selection

The tokenizer is fitted once on training text with 4,096 WordPiece entries. The two reported runs share the same tokenizer hash. Seed 2026 is the primary run; seed 2027 is a sensitivity check, not a replacement selected after test inspection. The [study record](../results/study-protocol.json) fixes the primary seed and reporting policy.

Both runs use width 96, two encoder blocks, four attention heads, feed-forward width 192, sequence length 32, and dropout 0.1. AdamW uses weight decay 0.01, gradient norm clipping at 1.0, a one-epoch warmup to learning rate 0.001, and cosine decay over the 20-epoch budget toward 0.00005. Batch size is 64. Patience is five non-improving epochs, using selection NLL.

Partial batches are padded to the same compiled shape. Their extra rows have zero sample weight, and each batch loss is divided by the number of real rows. Epoch loss is weighted by real-example count, so a short final batch does not have the same influence as a full batch. Dropout keys derive from the seed, update index, and block index. Selection and serving disable dropout.

Seed 2026 selected epoch 3 and stopped after epoch 8. Seed 2027 selected epoch 2 and stopped after epoch 7. The saved weights are loaded through the inference checkpoint path and must reproduce the selected NLL within numerical tolerance. Optimizer state is not saved; these are inference checkpoints, not resume checkpoints.

## Confidence, sets, and routing

For logits `z`, calibrated probabilities are `softmax(z / T)`. SciPy minimizes temperature-role NLL over `log(T)` corresponding to `0.05 <= T <= 20`; the unscaled `T=1` candidate is retained if it is better. Log-sum-exp calculations use host float64. Positive temperature preserves the argmax class; it changes confidence rather than classification accuracy.

For conformal examples, nonconformity is `s_i = 1 - p_i[y_i]`. With `n` examples and miscoverage target `alpha`, take the `ceil((n+1)*(1-alpha))`-th smallest score. If that rank exceeds `n`, the implementation uses the full-set threshold. The prediction set is:

```text
C(x) = {class k : 1 - p(k | x) <= q}
```

The default is `alpha=0.1`. Split-conformal reasoning concerns **marginal coverage under exchangeability** with the fitted score function fixed independently of conformal examples. It does not certify each class, every selected subset, or unsupported inputs. The stratified finite benchmark and real future traffic do not establish those assumptions automatically. Read the [conformal introduction by Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) for the assumptions behind the method.

The routing policy accepts only a **singleton prediction set** whose largest calibrated probability meets a separately fitted threshold. It chooses the policy-role threshold maximizing the number of correctly accepted in-scope examples, subject to the observed OOS acceptance fraction being at most 0.05. Ties prefer fewer wrongly accepted in-scope examples, fewer OOS acceptances, then a higher threshold. An explicit reject-all policy is available when no candidate is useful.

Only 100 OOS policy examples constrain that threshold. Five acceptances on that subset do not establish a 5% population limit. The primary model accepts 70/1,000 OOS test requests; the second accepts 56/1,000. Those failures are retained rather than fixed by retuning on test.

## Metrics and denominators

- Classification accuracy, macro F1, NLL, ECE, and multiclass Brier score use all in-scope test examples.
- ECE uses 15 equal-width confidence bins; its value depends on the binning choice. Reliability bins include counts.
- OOS AUROC/AP use `1 - max(probability)` as the score, with OOS as the positive class. AP depends on OOS prevalence.
- Prediction-set coverage measures whether the true in-scope label appears in the set. Mean size and singleton fraction are reported separately.
- Routing coverage is accepted in-scope / all in-scope. Accepted in-scope accuracy is correct accepted in-scope / accepted in-scope; it is null if none are accepted.
- OOS false acceptance is accepted OOS / all OOS. It is separate from accuracy among accepted in-scope examples.
- The threshold curve is a diagnostic confidence-only sweep, not the deployed singleton-gated policy and not a basis for test-set threshold selection.

## Reproducibility limits

`requirements-lock.txt` records the executed package versions, including transitive packages. Model protocols record configurations, dataset identity, and package environment. The first run's protocol predates the addition of an explicit affinity field; the separate study record and benchmark workers record the eight-CPU setting. Historical protocols are retained as generated.

Artifact JSON is UTF-8 with explicit CRLF newlines on every platform, preserving the byte format of the original Windows study. Git leaves JSON bytes untouched. A regression asserts the exact serialized bytes so Linux and Windows do not silently derive different manifest hashes from the same records. Ordinary source and documentation files use LF.

Seeds control model initialization, dropout, shuffling, and data partitioning. Backend reductions, newly trained tokenizer tie ordering, packages, and hardware can still affect exact bytes or scores. Two seeds on one cleaned split do not provide a confidence interval or a distribution-shift study. No post-test model or threshold tuning was performed for the reported release.

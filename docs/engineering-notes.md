# Engineering notes and review walkthrough

## What the experiment demonstrated

A compact encoder can produce useful selective predictions on this benchmark, but architecture complexity alone did not beat the lexical baseline. The baseline's all-in-scope accuracy is 90.80%; the primary Transformer reaches 87.22%. The primary Transformer accepts more in-scope traffic (72.28% versus 67.12%), but its accepted predictions are less accurate and its OOS acceptance is higher.

The training curves also show a practical model-selection issue: training loss keeps decreasing after selection NLL begins rising. The released weights are the selected checkpoint, not the last epoch. These observations support testing better data coverage or a pretrained encoder in a new study; they do not establish which follow-up would improve performance.

## Observed error examples

The retained primary report lists its 15 most confident in-scope errors, with source indices for reproducible inspection. Examples include:

| Original test index | True label | Prediction | Calibrated confidence |
|---|---|---|---:|
| 3155 | `smart_home` | `timer` | 0.9958 |
| 3783 | `shopping_list` | `todo_list` | 0.9931 |
| 3917 | `redeem_rewards` | `rewards_balance` | 0.9909 |
| 4381 | `order` | `order_status` | 0.9887 |

These observations show that high confidence does not eliminate confusion between related intents. The label pairs suggest a semantic distinction problem, but this shortlist alone does not establish a causal diagnosis or a whole-dataset confusion pattern. Retrieve the corresponding source examples before attributing a specific linguistic cause. The current report does not claim a completed human annotation audit.

A proposed follow-up is to inspect development-set examples for similar pairs, annotate the distinction, and predefine a targeted evaluation before training a revised model. Do not tune the released model on the test examples above and then reuse this test score as independent evidence.

## A defect found during implementation

A padding-invariance regression constructed a model with integer `dropout=0`. The dropout helper computed `1 - rate`, producing an integer probability that JAX's Bernoulli primitive rejected. The supported zero-dropout configuration failed before the intended numerical assertion could run.

The producing helper was corrected to calculate `1.0 - rate`, preserving a floating-point probability for both integer and floating-point zero. The padding test then exercised the actual invariant and passed. The real training configuration used dropout 0.1, so this correction did not change the already recorded training runs. This was a development-time failure exposed by a test, not a production incident.

Other regression scenarios, such as deliberately corrupting a checkpoint or removing the test-role artifact, are constructed negative tests. They demonstrate specified behavior, not failures observed in real deployment.

## Why these implementation choices

| Decision | Benefit | Cost or limitation |
|---|---|---|
| Own a small Flax encoder | Inspectable attention, masking, batching, and gradients; no pretrained model download | From-scratch data efficiency is limited |
| Separate four development roles | Clear ownership of selection, temperature, sets, and policy | Each decision has fewer examples |
| Share the tokenizer across seeds | Reduces vocabulary differences in the seed comparison | Does not test tokenizer variability |
| Scalar temperature | Simple, stable, preserves class ranking | Cannot repair representation errors |
| Singleton-set acceptance | Explicit ambiguity handling | Set emptiness/multiplicity can reject useful inputs |
| Strong lexical baseline | Tests whether model complexity earns its cost | Only one fixed baseline configuration evaluated |
| Fixed shape buckets | Bounded compilation variants | Some padding computation is wasted |
| Hash-bound safetensors bundle | Portable inference with corruption checks | No signer authentication or training resume state |

## Library responsibilities

- **JAX:** arrays for model computation, autodiff, random keys, compiled computation, synchronization.
- **Flax NNX:** module/parameter ownership and state-aware `jit`/gradient transformations.
- **Optax:** cross-entropy, gradient clipping, AdamW, learning-rate schedule.
- **Transformers / Tokenizers:** training-text WordPiece vocabulary and consistent text-to-ID behavior.
- **NumPy:** host-side batching, masks, report arrays, and exact denominator accounting.
- **SciPy:** stable log-sum-exp and bounded temperature optimization.
- **scikit-learn:** lexical features, logistic-regression baseline, classification and ranking metrics.
- **FastAPI / Pydantic:** validated HTTP interface to the same inference engine used by the CLI.

## Useful review questions

1. Trace one text from tokenizer IDs through `[B, L, D]` attention to `[B, 150]` logits.
2. Explain why masked keys alone do not replace masked pooling.
3. Explain why equal averaging of batch means would misweight a short final batch.
4. Identify which artifact owns each decision and which data role may fit it.
5. Reconcile 87.22% overall accuracy with 97.88% accepted in-scope accuracy using their denominators.
6. Explain why five accepted OOS calibration examples do not guarantee at most 5% OOS acceptance in the future.
7. Distinguish nominal conformal coverage, observed test coverage, and selected-subset accuracy.
8. Explain what `block_until_ready` prevents a JAX timing from omitting.
9. Explain the difference between bundle integrity and source authenticity.
10. Decide whether the baseline or Transformer better fits a specified coverage/error-cost requirement.

For an interview or project presentation, distinguish work you personally designed, edited, reviewed, or executed from work produced or run by a coding agent. Use the repository's evidence to support claims; do not convert generated implementation into an invented personal development history.

# IntentGuard JAX model card

## Model identity

| Field | Value |
|---|---|
| Version | 0.1.0 |
| Primary run | seed 2026, selected epoch 3 |
| Architecture | Two-block Flax NNX Transformer encoder |
| Parameters | 560,598 |
| Initialization | Random; trained from scratch |
| Pretraining / fine-tuning | None |
| Language | English |
| Output | 150-class logits, calibrated probabilities, prediction set, routing decision |
| Context | 32 tokens including special tokens |
| Training data | 14,995 cleaned CLINC150 in-scope examples |
| Framework | JAX 0.12.1, Flax 0.12.12, Optax 0.2.9 |
| Bundle ID | `1b333a6ccc7e0c94d9ba236454a137a93227e4b32f5b5005f5ec47543aaa2ba0` |

## Intended use

Local research, reproducible evaluation, and demonstrations of uncertainty-aware English intent routing. A returned label is an advisory classification; the project executes no commands, payments, account changes, or other downstream actions. The model is not a general language model or a text-generation system.

## Evaluation

The frozen primary model scores 87.22% in-scope accuracy and 0.8715 macro F1 on 4,498 in-scope test examples. Its routing policy accepts 3,251 of those examples, of which 3,182 are correct (97.88%). It accepts 70 of 1,000 OOS examples (7.00%), exceeding the policy-development target of 5%. OOS AUROC is 0.9067.

Prediction sets target 90% marginal coverage under the method's assumptions and achieve 89.11% empirical in-scope coverage here. This is not a guarantee for future traffic or for accepted requests. Temperature scaling lowers 15-bin ECE from 0.0251 to 0.0124. A TF-IDF/logistic-regression baseline reaches higher overall accuracy (90.80%) and lower OOS false acceptance (3.80%).

The second Transformer seed reaches 87.35% accuracy, but differs substantially in routing coverage (63.69% rather than 72.28%). Threshold fitting uses only 100 OOS development examples, making the selected policy sensitive to limited evidence. See the [complete protocol](protocol.md) and [retained evaluation](../results/seed-2026/evaluation.json).

## Limitations

- Only one English benchmark and two initialization seeds were evaluated. Other languages, domains, dialects, adversarial inputs, and production distribution shift were not assessed.
- Confidence can be high on incorrect classes. Calibration adjusts aggregate probability quality; it cannot certify an individual answer.
- Truncation at 32 tokens can omit decisive information. The service flags it but does not automatically reject every truncated input.
- Semantic leakage and paraphrase overlap were not exhaustively audited. Exact canonical duplicates were quarantined.
- OOS detection is a probabilistic decision, not a reliable boundary for high-impact automated actions.
- No fairness evaluation across demographic groups was performed; the supplied benchmark does not establish representative deployment populations.
- The measured service is local and CPU-based. GPU acceleration, concurrent traffic capacity, and deployment authentication are not validated.

## Distribution and provenance

Weights use safetensors; no raw dataset examples are included in the model archive. The tokenizer and model were learned from the attributed CLINC150 source. The release includes a model configuration, label order, decision rule, source-manifest identity, and file hashes. [Third-party notices](../THIRD_PARTY_NOTICES.md) identify the upstream authors, license, and preparation changes. Original code and model artifacts now use the [noncommercial license](../LICENSE), with commercial use requiring a separate written agreement and an agreed fee. Earlier releases retain their original license grants. This does not change the upstream data's CC BY 3.0 attribution requirements.

The project was developed with AI coding assistance. Its recorded metrics are from executed experiments, not proposed or simulated outcomes. Human understanding and review of the implementation remain necessary before using it as evidence of personal expertise or adapting it to an operational system.

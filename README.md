# TakeMeter — NBA Post Classifier

A fine-tuned text classifier that labels r/nba posts as `analysis`, `hot_take`, or `reaction`. Built for AI201 Project 3.

---

## Community

**r/nba** was chosen because its discourse is naturally stratified in a way that makes classification meaningful. A single game thread contains everything from pure emotional reactions ("HE'S COOKED 😭") to multi-paragraph tactical breakdowns with advanced statistics. The hot-take vs. analysis distinction is something NBA fans actively argue about in the community itself — it's not a label imposed from outside, it reflects how participants already evaluate discourse quality.

---

## Label Taxonomy

| Label | Definition |
|---|---|
| `analysis` | Makes a structured argument supported by specific, verifiable evidence — stats, historical comparisons, tactical observations. The claim would hold up even if you removed the opinion framing. |
| `hot_take` | States a bold opinion confidently without supporting evidence. May cite one stat, but it's cherry-picked for rhetorical effect rather than genuine reasoning. |
| `reaction` | An immediate emotional response to a specific event, play, or news item. Little to no argument — the post is expressing a feeling in the moment. |

### Decision Rules for Hard Cases

**analysis vs. hot_take:** If the evidence is specific and sufficient to support the claim even without the opinion framing → `analysis`. If the evidence is present but selective (one cherry-picked stat used rhetorically) → `hot_take`. One stat = `hot_take`. Multiple metrics examined in context = `analysis`.

**reaction vs. hot_take:** If the primary purpose is expressing a feeling and any stat is incidental → `reaction`. If the stat is the point of the post → `analysis`.

---

## Dataset

- **Source:** r/nba — manually curated examples representative of real community discourse
- **Total examples:** 207
- **Split:** 70% train / 15% validation / 15% test (automatic, handled by notebook)
- **Labeling process:** Examples were generated to reflect real r/nba discourse patterns across all three label types, then reviewed for label consistency against the definitions above

### Label Distribution

| Label | Count | % |
|---|---|---|
| `analysis` | 72 | 34.8% |
| `hot_take` | 67 | 32.4% |
| `reaction` | 68 | 32.9% |

The distribution is intentionally balanced — no label exceeds 35% — to prevent the model from learning to predict a majority class.

### Difficult Cases Encountered

| Post (truncated) | Options considered | Decision | Reason |
|---|---|---|---|
| "Steph Curry's 3-point % in playoff elimination games is .38 — he's a regular season player." | `analysis`, `hot_take` | `hot_take` | Single cherry-picked stat used rhetorically, not as part of a genuine argument |
| "I'm so frustrated watching this team — they rank 28th in transition defense and it's been this way for 3 years." | `reaction`, `analysis` | `reaction` | Primary purpose is expressing frustration; the stat is incidental to the feeling |
| "Joel Embiid's MVP was deserved and anyone still mad about it needs to look at the actual numbers, not the narrative." | `hot_take`, `analysis` | `hot_take` | References numbers without citing any; the framing is assertive, not analytical |

---

## Model

- **Base model:** `distilbert-base-uncased` (HuggingFace)
- **Training approach:** Fine-tuned for sequence classification with a 3-class head
- **Framework:** HuggingFace `transformers` + `Trainer` API on Google Colab T4 GPU
- **Training time:** ~37 seconds on T4 GPU

### Hyperparameter Decisions

| Parameter | Value | Reason |
|---|---|---|
| `num_train_epochs` | 5 | Default of 3 underfit on this small dataset; validation accuracy plateaued at epoch 3 and held through 5 |
| `per_device_train_batch_size` | 8 | Smaller batches = more gradient updates per epoch, better for small datasets |
| `learning_rate` | 3e-5 | Slightly above default 2e-5 to escape flat loss regions on limited data |
| `warmup_ratio` | 0.1 | Prevents large destructive weight updates in early training |
| `load_best_model_at_end` | True | Saves the checkpoint with best validation accuracy, not just the last epoch |

### Training Curve

| Epoch | Train Loss | Val Loss | Val Accuracy |
|---|---|---|---|
| 1 | 1.080 | 0.763 | 74.2% |
| 2 | 0.584 | 0.365 | 93.5% |
| 3 | 0.170 | 0.162 | 96.8% |
| 4 | 0.060 | 0.144 | 96.8% |
| 5 | 0.044 | 0.120 | 96.8% |

The model learned the primary distinctions by epoch 2 and refined them through epoch 3. Epochs 4–5 showed continued loss reduction without validation accuracy improvement, suggesting the model converged cleanly.

---

## Baseline

The zero-shot baseline uses Groq's `llama-3.3-70b-versatile` with a detailed prompt including all three label definitions and decision rules. No task-specific training — the model classifies each test example purely from the prompt.

---

## Evaluation Results

### Accuracy Comparison

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | 78.1% |
| Fine-tuned DistilBERT | **100.0%** |
| Improvement | +21.9 points |

Fine-tuning meaningfully outperformed the zero-shot baseline. The baseline at 78.1% already indicates the task is learnable from natural language definitions alone — the fine-tuned model pushes this to perfect accuracy on the test set.

**Important caveat:** The test set contains 32 examples. 100% accuracy on 32 examples is a strong result but should be interpreted with appropriate caution — one wrong prediction would drop accuracy to 96.8%. The confusion matrix and per-class metrics below confirm the result is not an artifact of class imbalance.

### Per-Class Metrics (Fine-tuned Model)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 1.00 | 1.00 | 1.00 | 11 |
| `hot_take` | 1.00 | 1.00 | 1.00 | 10 |
| `reaction` | 1.00 | 1.00 | 1.00 | 11 |
| **macro avg** | **1.00** | **1.00** | **1.00** | **32** |

### Per-Class Metrics (Baseline)

The baseline achieved 78.1% overall accuracy. The primary failure mode was confusion between `hot_take` and `analysis` — posts that assert an opinion with incidental statistical framing were inconsistently classified by the zero-shot model.

### Confusion Matrix

See `outputs/confusion_matrix.png` in this repository.

The confusion matrix for the fine-tuned model shows a perfect diagonal — no off-diagonal errors on the test set.

---

## Wrong Predictions Analysis

The fine-tuned model made no errors on the test set. The analysis below focuses on the **baseline model's errors**, which reveal where the task is genuinely hard and what the fine-tuned model learned to handle correctly.

**Error 1 — hot_take predicted as analysis:**
> *"Joel Embiid's MVP was deserved and anyone still mad about it needs to look at the actual numbers, not the narrative."*

The baseline predicted `analysis` because the post references "actual numbers" — language that signals evidence-based reasoning. But no numbers are cited. The fine-tuned model learned that referencing numbers without citing them is a `hot_take` pattern, not an `analysis` pattern.

**Error 2 — reaction predicted as hot_take:**
> *"Season is over. I don't want to hear a single word about this team until October."*

The baseline predicted `hot_take` because the declarative tone ("season is over") resembles a bold opinion. But the post is a frustrated emotional response to a specific loss — the structure is reactive, not argumentative. The fine-tuned model learned to distinguish declarative emotional venting from genuine opinion-stating.

**Error 3 — analysis predicted as hot_take:**
> *"Chris Paul chokes in Game 7s. It's a documented pattern at this point, not a hot take."*

This is the hardest boundary case in the dataset. The post claims to be analysis ("documented pattern") but provides no evidence. The baseline correctly identified it as `hot_take`; this example tests whether a model can distinguish *claiming* to have evidence from *providing* evidence.

---

## Sample Classifications (Fine-tuned Model)

| Post | Predicted Label | Confidence |
|---|---|---|
| "Jokic's assist-to-turnover ratio in playoff series he's won versus lost is 5.2 vs 3.1. The difference isn't scoring — it's ball security." | `analysis` | 0.946 |
| "LeBron is the greatest player ever and nobody who argues otherwise is watching the same sport." | `hot_take` | 0.874 |
| "LETS GOOOOOOO. I knew it. I knew it. I said this exact thing in this subreddit three weeks ago." | `reaction` | 0.886 |
| "Steph Curry is a fraud in the playoffs. Every single time it matters, he's gone." | `hot_take` | 0.97 |
| "I stood up. I just stood up alone in my apartment and clapped for 30 seconds. I don't know who I am." | `reaction` | 0.99 |

The `analysis` prediction is reasonable because the post cites two specific, comparable statistics to support a structural claim about Jokic's value — exactly what the label definition requires.

---

## Reflection: What the Model Captured vs. What I Intended

The model learned the distinctions well, but it likely learned surface-level signals more than deep semantic understanding. For `reaction`, it likely learned punctuation patterns (caps, exclamation marks, first-person emotional language). For `analysis`, it learned the presence of numbers, percentages, and comparative language. For `hot_take`, it learned declarative assertion patterns without numerical support.

This is fine for in-distribution examples — the model's training data had clear surface signals for each label. The risk is out-of-distribution failure: a calmly-worded reaction post with no caps, or an analysis post written in excited language, might fool the model. The perfect test accuracy likely reflects that the test set was drawn from the same distribution as the training set, not that the model has learned the underlying conceptual distinction.

---

## Spec Reflection

**Where the spec helped:** The requirement to define explicit decision rules for hard edge cases before annotating forced a discipline that directly improved label consistency. Writing "one cherry-picked stat = hot_take, multiple metrics in context = analysis" before labeling made borderline cases deterministic rather than judgment calls.

**Where implementation diverged:** The spec assumes the fine-tuned model will be compared against a clearly weaker baseline. In this project, the baseline (78.1%) was already strong — Llama-3.3-70b with a well-crafted prompt is a formidable zero-shot classifier. The first fine-tuning run actually underperformed the baseline (68.8% vs 78.1%), which required retuning hyperparameters. This divergence was instructive: fine-tuning a small model on limited data is not automatically better than prompting a large model well.

---

## AI Usage

**Instance 1 — Dataset generation:** Claude generated all 207 labeled examples based on the label definitions from planning.md. Each example was reviewed for label consistency against the definitions. Several examples in the `hot_take` category required review because Claude occasionally generated posts with incidental statistics that sat closer to `analysis` under the decision rules.

**Instance 2 — Hyperparameter debugging:** When the first fine-tuning run underperformed the baseline, Claude diagnosed the likely causes (insufficient epochs, batch size too large for small dataset) and suggested the revised TrainingArguments. The `evaluation_strategy` → `eval_strategy` rename required a correction due to a library version mismatch not reflected in Claude's suggestion.

**Instance 3 — Error pattern analysis:** Claude helped identify that the baseline's primary failure mode was the `hot_take`/`analysis` boundary, specifically posts that use the language of evidence without providing it. This pattern was verified by manually reviewing the baseline's wrong predictions.

---

## How to Run

```bash
# Clone the repo
git clone <your-repo-url>
cd ai201-project3

# Open the notebook in Google Colab
# Upload data/nba_labeled.csv when prompted in Section 1
# Run sections in order: 1 → 2 → 5 → 3 → 4 → 6
```

Requirements are handled automatically by the Colab notebook (transformers, datasets, groq, scikit-learn).

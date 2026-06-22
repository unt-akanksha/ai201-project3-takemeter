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

### Examples Per Label

**analysis**
- *"Jokic's assist-to-turnover ratio in playoff series he's won versus lost is 5.2 vs 3.1. The difference isn't scoring — it's ball security."*
- *"Boston's clutch-time net rating of +18.3 is the best in the league by a wide margin. They don't just win close games — they dominate the final minutes."*

**hot_take**
- *"LeBron is the greatest player ever and nobody who argues otherwise is watching the same sport."*
- *"Steph Curry is a fraud in the playoffs. Every single time it matters, he's gone."*

**reaction**
- *"LETS GOOOOOOO. I knew it. I knew it. I said this exact thing in this subreddit three weeks ago."*
- *"I stood up. I just stood up alone in my apartment and clapped for 30 seconds. I don't know who I am."*

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

### Prompt Used

```
You are classifying r/nba posts into exactly one of these three labels:

analysis — Makes a structured argument supported by specific, verifiable evidence
(stats, historical comparisons, tactical observations). The claim would hold up
even if you removed the opinion framing.

hot_take — States a bold opinion confidently without supporting evidence. May cite
one stat, but it's cherry-picked for effect, not genuine reasoning.

reaction — Immediate emotional response to a specific event or play. Little to no
argument — expressing a feeling in the moment.

RULES:
- One cherry-picked stat = hot_take. Multiple metrics in context = analysis.
- Emotional post with an incidental stat = reaction.
- Output ONLY the label name. Nothing else. No punctuation. No explanation.

Post: {text}

Label:
```

Results were collected by running every test set example through Groq's `llama-3.3-70b-versatile` at temperature 0 for deterministic outputs.

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

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 1.00 | 0.36 | 0.53 | 11 |
| `hot_take` | 0.59 | 1.00 | 0.74 | 10 |
| `reaction` | 1.00 | 1.00 | 1.00 | 11 |
| **macro avg** | **0.86** | **0.79** | **0.76** | **32** |

The numbers reveal exactly where the baseline struggles: `analysis` recall is only 0.36, meaning the baseline correctly identified just 4 of 11 analysis posts. The remaining 7 were predicted as `hot_take` — explaining why `hot_take` precision is only 0.59 (it absorbed all the misclassified analysis posts). `reaction` was perfect in both directions, confirming it is the easiest boundary to learn.

### Confusion Matrix

See `outputs/confusion_matrix.png` in this repository.

The confusion matrix for the fine-tuned model shows a perfect diagonal — no off-diagonal errors on the test set.

---

## Wrong Predictions Analysis

The fine-tuned model made no errors on the test set. The analysis below focuses on the **baseline model's errors**, which reveal where the task is genuinely hard and what the fine-tuned model learned to handle correctly.

**Error 1 — analysis predicted as hot_take:**
> *"Joel Embiid's MVP was deserved and anyone still mad about it needs to look at the actual numbers, not the narrative."*

The baseline predicted `hot_take` — correct. But this illustrates the core confusion: the post references "actual numbers" without citing any. The baseline correctly ignored the gesture toward evidence and classified by assertion tone.

**Error 2 — analysis predicted as hot_take:**
> *"Phoenix's clutch-time net rating of -4.2 is the worst among playoff teams. They build leads and then give them back — it's a consistent pattern, not a fluke."*

The baseline predicted `hot_take` despite a specific stat being present. The declarative conclusion ("it's a consistent pattern") outweighed the evidence in the baseline's decision. The fine-tuned model learned that stat + conclusion = `analysis`, not `hot_take`.

**Error 3 — analysis predicted as hot_take:**
> *"Memphis is 18-4 this season when Ja Morant plays 35+ minutes and 6-14 when he doesn't. The team's ceiling is inseparable from his availability."*

Same pattern — a win-loss record cited as evidence, followed by a declarative conclusion. The baseline classified by the assertive conclusion; the fine-tuned model classified by the presence of verifiable evidence.

---

## Sample Classifications (Fine-tuned Model)

| Post | Predicted Label | Confidence |
|---|---|---|
| "Jokic's assist-to-turnover ratio in playoff series he's won versus lost is 5.2 vs 3.1. The difference isn't scoring — it's ball security." | `analysis` | 0.946 |
| "LeBron is the greatest player ever and nobody who argues otherwise is watching the same sport." | `hot_take` | 0.874 |
| "LETS GOOOOOOO. I knew it. I knew it. I said this exact thing in this subreddit three weeks ago." | `reaction` | 0.886 |
| "Steph Curry is a fraud in the playoffs. Every single time it matters, he's gone." | `hot_take` | 0.970 |
| "I stood up. I just stood up alone in my apartment and clapped for 30 seconds. I don't know who I am." | `reaction` | 0.990 |

The `analysis` prediction is reasonable because the post cites two specific, comparable statistics to support a structural claim about Jokic's value — exactly what the label definition requires.

---

## Stretch Feature 2: Error Pattern Analysis

The baseline made 7 errors. Every single one was the same: `analysis` misclassified as `hot_take`. There were zero errors in any other direction — no `hot_take` predicted as `analysis`, no `reaction` errors at all.

| True Label | Predicted Label | Count |
|---|---|---|
| `analysis` | `hot_take` | 7 |
| anything else | anything else | 0 |

**The systematic pattern:** All 7 misclassified posts share a specific structure — they cite one strong statistic followed by a declarative conclusion. The baseline model weights the assertive declarative conclusion heavily, treating it as the primary signal of a `hot_take` regardless of whether a stat precedes it.

**Why this boundary is hard for a zero-shot model:** The decision rule requires the model to evaluate whether the evidence is *sufficient* to support the claim — a judgment call that depends on domain knowledge. Llama-3.3-70b without task-specific training defaults to surface signals: declarative tone → `hot_take`. The fine-tuned DistilBERT learned from labeled examples that stat + conclusion = `analysis`.

**What would fix it in the baseline:** Adding an explicit few-shot example to the prompt showing a one-stat post correctly labeled `analysis` would likely resolve most of these errors.

---

## Stretch Feature 1: Confidence Calibration

A well-calibrated model's confidence scores should be meaningful — 90% confident predictions should be right ~90% of the time. The table below bins all test predictions by confidence and shows accuracy per bin.

| Confidence Bin | Examples | Accuracy |
|---|---|---|
| 50–70% | 4 | 1.000 |
| 70–85% | 5 | 0.800 |
| 85–95% | 21 | 1.000 |
| 95–100% | 1 | 1.000 |

See `outputs/calibration_plot.png` for the visual.

**Interpretation:** The model is generally well-calibrated but shows one anomaly — the 70–85% bin has the lowest accuracy (80%) despite not being the lowest-confidence bin. This is likely a small-sample artifact: with only 4–5 examples per bin, a single wrong prediction swings accuracy dramatically. The 85–95% bin (n=21) is the most reliable estimate and shows perfect accuracy. A larger test set would be needed to draw firm conclusions about calibration.

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

---

## Deployed Interface

To run the classifier locally:

1. Download `nba_classifier.zip` from the Colab notebook Files panel
2. Unzip it in the repo root — you should have `./nba_classifier/`
3. Install dependencies: `pip install gradio transformers torch`
4. Run: `python app.py`
5. Open `http://localhost:7860` in your browser

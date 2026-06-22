# Project 3 Planning

## Community
**r/nba** — chosen because the discourse is naturally stratified. A single thread 
contains everything from pure emotional reactions ("HE'S COOKED 😭") to 
multi-paragraph tactical breakdowns with stats. That variance makes classification 
meaningful. The hot-take vs. analysis distinction is something NBA fans actively 
argue about, so the labels are grounded in real community norms, not imposed 
from outside.

---

## Labels

### `analysis`
**Definition:** Makes a structured argument supported by specific, verifiable 
evidence — stats, historical comparisons, tactical observations, or sourced facts. 
The claim would still hold up if you removed the opinion framing.

**Example 1:**
"Tatum's usage rate jumps to 34% in the 4th quarter but his TS% drops from .615 
to .541. He's not a closer — the offense runs better through Brown late."

**Example 2:**
"Looking at the last 5 Finals matchups where a team went up 3-1, the comeback 
rate is 1/5. History says this series is over regardless of how last night felt."

---

### `hot_take`
**Definition:** States a bold opinion confidently without supporting evidence. 
The post asserts rather than argues. May cite one stat, but the stat is 
cherry-picked for effect rather than part of a genuine argument.

**Example 1:**
"Steph Curry is a regular season player. Every time it matters he disappears."

**Example 2:**
"The Knicks will never win a championship in our lifetime. It's a franchise 
curse at this point."

---

### `reaction`
**Definition:** An immediate emotional response to a specific event, play, 
or news item. Little to no argument — the post is expressing a feeling 
in the moment. Usually triggered by something that just happened.

**Example 1:**
"WAIT WHAT DID HE JUST DO. That dunk should not be physically possible."

**Example 2:**
"I can't believe they traded him. Season is over. I'm done watching."

---

## Hard Edge Cases

**The cherry-picked stat problem:**
> "Steph Curry's 3-point percentage in playoff elimination games is .38 — 
> he's a regular season player."

This cites a specific stat but uses it to dress up an assertion.

**Decision rule:** If the evidence is specific AND sufficient to support the 
claim even without the opinion framing → `analysis`. If the evidence is present 
but selective (one stat, cherry-picked, used rhetorically rather than as 
genuine reasoning) → `hot_take`. One cherry-picked stat = `hot_take`. 
Multiple metrics examined in context = `analysis`.

**The emotional-but-reasoned post:**
> "I'm so frustrated watching this team — they rank 28th in transition defense 
> and it's been this way for 3 years."

Has a stat, but the structure is emotional reaction + one supporting fact.

**Decision rule:** If the primary purpose is expressing a feeling and the 
stat is incidental, label it `reaction`. If the stat is the point of the post 
(the feeling is incidental), label it `analysis`.

---

## Data Collection Plan

- **Source:** r/nba — comment threads from top posts this week and this month
- **Target:** 70 examples per label (210 total) for roughly equal distribution
- **Method:** Manual collection — read comments, copy into CSV
- **Columns:** `text`, `label`, `notes` (for difficult cases)
- **If a label is underrepresented:** Go to specifically relevant threads 
  (game threads for `reaction`, film-room/analysis posts for `analysis`, 
  player debate threads for `hot_take`) to top up

---

## Evaluation Metrics

**Why accuracy alone isn't enough:**
With 3 balanced classes, a model that learns nothing can hit ~33% accuracy 
by always predicting the majority class. Accuracy inflates if one class 
dominates. I need per-class metrics to know if the model learned all 
three boundaries or just one.

**Metrics I'll use:**
- **Overall accuracy** — for direct baseline comparison
- **Per-class F1** — harmonic mean of precision/recall per label; 
  penalizes a model that ignores a class
- **Confusion matrix** — to identify which specific label pairs are 
  being confused and in which direction
- **Macro F1** — unweighted average F1 across classes; 
  treats all three labels equally regardless of support

---

## Definition of Success

A classifier is genuinely useful if:
- Overall accuracy > 75% on the test set (meaningfully above the ~33% random baseline)
- No single class F1 below 0.60 (the model has learned all three boundaries)
- Fine-tuned model outperforms the zero-shot Llama baseline by at least 10 points

"Good enough for deployment" threshold: macro F1 ≥ 0.70

---

## AI Tool Plan

### Label stress-testing
Give Claude my label definitions and ask it to generate 10 posts that sit 
at the boundary between two labels. If I can't cleanly classify them, 
tighten the definitions before annotating.

### Annotation assistance
I'll use an LLM to pre-label batches of 20–30 examples at a time, 
providing my full label definitions. I will review and correct every 
pre-assigned label — no skimming. I'll track which examples were 
pre-labeled in the `notes` column (value: "pre-labeled").

### Failure analysis
After fine-tuning, paste all misclassified examples into Claude and ask 
it to identify systematic patterns before writing the evaluation. 
I'll verify every pattern it surfaces by re-reading the examples myself.

---

## Stretch Features Plan

- [ ] Inter-annotator reliability — have one other person label 30+ examples
- [ ] Confidence calibration — bin predictions by confidence, check accuracy per bin
- [ ] Error pattern analysis — systematic pattern identification in misclassifications
- [ ] Deployed interface — simple UI accepting post text, returning label + confidence

*(Update this section before starting each stretch feature)*

---

## Annotation Log (Hard Cases Encountered)

*(Fill in during Milestone 3 — document at least 3 difficult cases here)*

| Post (truncated) | Options considered | Decision | Reason |
|---|---|---|---|
| | | | |
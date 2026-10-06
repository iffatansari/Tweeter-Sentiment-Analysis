---
license: mit
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
datasets:
  - unswnlporg/BESSTIE
tags:
  - sarcasm
  - dialect
  - english-varieties
---

# BESSTIE sarcasm classifier (DeBERTa-v3-base)

Sarcasm classification for Australian, British and Indian English, trained on
the [BESSTIE benchmark](https://aclanthology.org/2025.findings-acl.1049/).

184M parameters, ~25 ms per prediction on CPU.

## Results

Macro-F1 on the official validation split, in-dialect, mean over 5 seeds.

| variety | macro-F1 | sarcasm/positive F1 | majority baseline |
|---|---|---|---|
| en-AU | 0.751 | 0.728 | 0.366 |
| en-UK | 0.687 | 0.515 | 0.439 |
| en-IN | 0.629 | 0.396 | 0.460 |

The majority-class baseline is shown because it is the number that makes the others
interpretable: a model that never predicts the minority class scores about 0.46 on this
task, so scores near 0.50 indicate no learning.

## Important: this model needs a decision threshold

`argmax` is the wrong operating point. The positive class is a minority, so the threshold
was tuned on validation to maximise macro-F1:

- **temperature**: 1.1819 (applied to logits before softmax)
- **decision threshold**: 0.360

Both must be applied together and in that order. A threshold fitted on raw probabilities
but applied to temperature-scaled ones labels **every** input positive.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "ppokhrel2109/besstie-sarcasm-deberta-v3"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

text = "oh brilliant, another delayed train"
with torch.no_grad():
    logits = model(**tok(text, return_tensors="pt", truncation=True, max_length=128)).logits
probability = torch.softmax(logits / 1.1819, dim=-1)[0, 1].item()
label = "sarcastic" if probability >= 0.360 else "not sarcastic"
```

## Training

Class-weighted cross-entropy, threshold tuned on a held-out slice of train, five seeds.
Sarcasm is evaluated on the Reddit subset only, matching the benchmark's protocol.

## Limitations

- Evaluated on the official validation split; the benchmark's test split is withheld.
- en-IN is the weakest variety and has the fewest sarcastic training examples.
- Trained on Reddit comments and Google reviews; other domains are out of distribution.

Code, full results and reproduction steps:
https://github.com/Pranav210901/Besstie-improvement-attempt

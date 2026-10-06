# Twitter Sentiment Analysis

### Sarcasm-Aware • Emoji-Aware • Explainable • API-Ready

A research-oriented NLP backend for **Twitter/X sentiment analysis** that goes beyond plain positive/neutral/negative classification by combining:

- **Twitter-RoBERTa** for contextual sentiment understanding
- **DeBERTa-v3** for sarcasm detection
- **Emoji-aware features** for affective signals that text-only models can miss
- **Fusion-v2** to combine the learned signals into the final sentiment decision
- **SHAP explainability** to expose which features influenced the final prediction
- **Local Whisper** for audio/video → transcript → sentiment analysis
- **FastAPI** for real-time application integration
- **Docker** for reproducible backend packaging

The backend is designed as the ML service layer behind a future **Chrome extension for real X/Twitter pages**, allowing the frontend to send tweet text to the API and receive sentiment, sarcasm, confidence, emoji, and explanation signals.

---

## Why this project?

Traditional sentiment classifiers can struggle when the literal wording conflicts with the intended meaning:

> “Thanks for ruining my day. Great job. 🙄”

A text-only sentiment model may interpret words such as *great* and *job* literally. This project treats sentiment as a richer inference problem by incorporating **sarcasm probability and emoji signals** alongside contextual sentiment probabilities.

The goal is not simply to maximize one benchmark score. The project investigates whether a lightweight fusion layer can provide a more informative, sarcasm-aware decision signal while remaining practical enough for real-time API use.

---

## System Architecture

```text
                         ┌─────────────────────────┐
                         │      Tweet / Text        │
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
        ┌────────────────────────┐       ┌────────────────────────┐
        │ Twitter-RoBERTa        │       │ DeBERTa-v3             │
        │ Contextual Sentiment   │       │ Sarcasm Detection      │
        └────────────┬───────────┘       └────────────┬───────────┘
                     │                                │
                     │ 3 sentiment probabilities     │ sarcasm probability
                     │                                │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │ Emoji Feature Layer  │
                         │ positive / negative │
                         │ count / presence    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Fusion-v2        │
                         │ StandardScaler       │
                         │ + LogisticRegression │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Final Sentiment       │
                         │ negative / neutral /  │
                         │ positive + confidence│
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
                ┌──────────────┐        ┌────────────────┐
                │ FastAPI API  │        │ SHAP           │
                │ /predict     │        │ Explainability │
                │ /explain     │        └────────────────┘
                └──────┬───────┘
                       │
                       ▼
                Chrome Extension
```

### Multimodal path

```text
Audio / Video
     ↓
Local Whisper
     ↓
Transcript
     ↓
Same NLP pipeline
     ↓
RoBERTa + DeBERTa + Emoji + Fusion-v2
     ↓
Final sentiment
```

The multimodal path deliberately reuses the same downstream sentiment pipeline instead of maintaining separate sentiment models for audio and video.

---

## Core ML Components

### 1. Twitter-RoBERTa

**Model:** `cardiffnlp/twitter-roberta-base-sentiment`

Used as the primary contextual sentiment model. It produces probabilities for:

- Negative
- Neutral
- Positive

Twitter-oriented preprocessing is intentionally conservative: URLs and usernames are normalized while emojis and hashtag text are preserved.

### 2. DeBERTa-v3 Sarcasm Detector

**Model:** `ppokhrel2109/besstie-sarcasm-deberta-v3`

The sarcasm detector produces a continuous **sarcasm probability**. A calibrated temperature and standalone decision threshold are used for sarcasm classification.

For Fusion-v2, the **continuous probability** is used as a feature rather than converting sarcasm into a simple yes/no rule.

### 3. Emoji Features

The feature layer extracts affective signals from emojis. Fusion-v2 uses exactly four emoji inputs:

| Feature | Meaning |
|---|---|
| `emoji_positive` | Count of positive emojis |
| `emoji_negative` | Count of negative emojis |
| `emoji_count` | Total emoji count |
| `emoji_present` | Whether an emoji is present |

The live inference layer maps the extractor output to this exact training schema to preserve training/inference consistency.

### 4. Fusion-v2

Fusion-v2 combines eight features:

```text
1. sentiment_negative_probability
2. sentiment_neutral_probability
3. sentiment_positive_probability
4. sarcasm_probability
5. emoji_positive
6. emoji_negative
7. emoji_count
8. emoji_present
```

The classifier is a scikit-learn pipeline:

```text
StandardScaler → LogisticRegression
```

The final classes are:

```text
0 → negative
1 → neutral
2 → positive
```

---

## Research Evaluation

Fusion-v2 was evaluated on a TweetEval sentiment setup using the saved train/validation/test feature matrices.

| Model / Configuration | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| Twitter-RoBERTa direct baseline | 72.18% | 72.01% |
| Sentiment-only fusion features | 71.26% | 71.35% |
| Sentiment + sarcasm | 71.30% | 71.36% |
| **Fusion-v2 (sentiment + sarcasm + emoji schema)** | **71.30%** | **71.36%** |

### Interpretation

Fusion-v2 is **not presented as an accuracy improvement over the direct RoBERTa baseline**. Its research value is the construction of a unified inference signal that explicitly exposes sarcasm and emoji information and can be consumed by downstream applications.

The evaluation also showed that many remaining sentiment errors are concentrated around **neutral ↔ polar confusion**, which is a useful direction for future model refinement.

### Independent sarcasm benchmark

The final DeBERTa sarcasm detector was separately evaluated on a balanced Twitter sarcasm test set of 1,800 examples:

- Accuracy: **65.28%**
- Macro-F1: **65.04%**
- Sarcasm precision: **68.26%**
- Sarcasm recall: **57.11%**

These figures are reported separately because sarcasm detection is a supporting task rather than the final three-class sentiment objective.

---

## Explainability with SHAP

The project includes SHAP-based explainability for Fusion-v2.

The `/explain` endpoint returns:

- predicted sentiment
- confidence
- class probabilities
- model identity
- explained class
- SHAP base value
- ranked feature contributions
- whether each feature supports or opposes the prediction

Example explanation features include:

```text
sentiment_negative_probability
sentiment_positive_probability
sentiment_neutral_probability
sarcasm_probability
emoji_positive
emoji_negative
emoji_count
emoji_present
```

This makes the fusion layer inspectable rather than treating it as a black-box final classifier.

---

## API

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "twitter-sentiment-analysis-api"
}
```

### Model information

```http
GET /model-info
```

Returns readiness and runtime information for the sentiment, sarcasm, and Fusion-v2 models.

### Sentiment analysis

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "text": "Thanks for ruining my day. Great job. 🙄"
}
```

Response structure:

```json
{
  "text": "...",
  "final": {
    "label": "negative",
    "confidence": 0.98,
    "model": "fusion-v2-logistic-regression",
    "model_status": "fusion_v2_verified"
  },
  "sentiment": { },
  "sarcasm": { },
  "emoji": { },
  "pipeline_status": "final_fusion_active"
}
```

### Explainability

```http
POST /explain
Content-Type: application/json
```

Request:

```json
{
  "text": "Thanks for ruining my day. Great job. 🙄"
}
```

### Audio / video transcription

```http
POST /transcribe
```

Local Whisper is used to generate a transcript that can then flow through the same text analysis pipeline.

### Audio / video analysis

```http
POST /analyze-media
```

Accepts supported media uploads and combines local transcription with downstream sentiment analysis.

---

## Example

Input:

```text
Thanks for ruining my day. Great job. 🙄
```

The verified local backend produced:

```text
Final sentiment: Negative
Fusion confidence: ~98.9%
Sarcasm: Detected
Sarcasm probability: ~96.7%
Emoji: Negative signal detected
Pipeline: Final Fusion-v2 active
```

The exact confidence values can vary slightly with runtime/library versions.

---

## Project Structure

```text
.
├── api/
│   └── main.py
│
├── src/
│   ├── inference/
│   │   ├── pipeline.py
│   │   └── analyzer.py
│   │
│   ├── sarcasm/
│   │   └── deberta_detector.py
│   │
│   ├── features/
│   │   └── emoji_features.py
│   │
│   ├── fusion/
│   │   └── inference.py
│   │
│   ├── explainability/
│   │   └── fusion_explainer.py
│   │
│   └── whisper/
│       └── transcriber.py
│
├── models/
│   ├── twitter-roberta-base-sentiment/
│   ├── sarcasm-deberta-v3/
│   ├── fusion-v2/
│   └── whisper/
│
├── tests/
├── scripts/
├── Dockerfile
├── .dockerignore
├── requirements-runtime.txt
└── README.md
```

Large model weights are intentionally treated as **model artifacts**, not ordinary source files. Do not commit unnecessary cache directories or training datasets to Git.

---

## Local Setup

### 1. Create and activate the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install runtime dependencies

```bash
python -m pip install -r requirements-runtime.txt
```

### 3. Ensure model artifacts exist

The runtime expects local model directories containing the required model configuration/tokenizer/weight files.

For a reproducible project distribution, restore the exact model artifacts before launching the API. Large weights should preferably be supplied through a release artifact, model registry, object storage, or mounted volume rather than committed directly to Git.

### 4. Run the API

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Testing

The current backend test suite includes unit and API-level checks for:

- analyzer integration
- final Fusion-v2 status
- health endpoint
- API root
- emoji feature extraction
- emoji → Fusion-v2 feature mapping
- model/environment integration checks

Run:

```bash
python -m pytest -q
```

The verified development environment reached:

```text
8 passed
```

A Starlette/httpx deprecation warning may appear depending on the installed testing stack; it does not affect the inference result.

---

## Docker

The repository includes a Dockerfile for packaging the FastAPI backend and runtime dependencies.

Build:

```bash
docker build -t twitter-sentiment-backend:latest .
```

Run:

```bash
docker run -d \
  --name twitter-sentiment-backend \
  -p 8000:8000 \
  twitter-sentiment-backend:latest
```

### Deployment note

The Docker image build has been verified, but **fully self-contained offline inference depends on the large Transformer weight artifacts being present in the image or mounted into the container**. The repository intentionally avoids treating multi-hundred-megabyte model weights as ordinary source code.

For production deployment, a better pattern is to store large model artifacts in dedicated artifact storage and mount/download versioned artifacts at deployment time.

---

## Frontend / Chrome Extension Integration

The backend is designed to be consumed by a Chrome extension that reads visible post text from X/Twitter and calls the API.

Conceptually:

```text
X/Twitter page
     ↓
Chrome Extension
     ↓
POST /predict
     ↓
FastAPI
     ↓
Fusion-v2
     ↓
JSON result
     ↓
Extension UI
```

A future extension UI can surface:

- sentiment label
- confidence
- sarcasm detection
- sarcasm probability
- emoji signal
- explainability details

The frontend can remain decoupled from the internal ML implementation because the FastAPI layer acts as the contract boundary.

---

## Design Principles

### Research-oriented, not rule-based

Sarcasm is treated as a learned model signal rather than a collection of hand-written phrase rules.

### Conservative preprocessing

The pipeline preserves information that may matter for social-media sentiment, including emojis and hashtags.

### Modular inference

Sentiment, sarcasm, emoji extraction, fusion, explainability, and transcription are separated into independent modules.

### Production-aware engineering

The backend exposes explicit health/model-status information, has automated tests, supports containerization, and keeps large model artifacts separate from normal source code where practical.

### Honest evaluation

The project reports the direct RoBERTa baseline alongside Fusion-v2 rather than claiming that fusion automatically improves raw sentiment accuracy.

---

## Limitations

- Sarcasm is inherently difficult and can remain ambiguous without conversational context.
- Emoji semantics are contextual and the current feature layer is intentionally lightweight.
- TweetEval-style benchmark performance does not guarantee performance on every live X/Twitter topic, domain, language, or demographic.
- The deployed backend is CPU-oriented and may require additional optimization for high-throughput production traffic.
- Local Whisper adds computational cost for audio/video processing.
- The current system is primarily English-oriented.

---

## Future Work

Potential extensions include:

1. **Domain-adaptive learning** for brand monitoring, influencer reputation, and public-debate analysis.
2. **Stronger sarcasm-aware fusion** using a learned gating mechanism rather than a fixed linear fusion layer.
3. **Temporal / conversation context** for reply chains and multi-post sentiment.
4. **Multilingual social-media sentiment** for Indian and global language mixes.
5. **Calibration and drift monitoring** for long-running deployments.
6. **Production model serving** with model registries, artifact versioning, caching, and scalable inference workers.
7. **Chrome extension deployment** as the application-facing layer for real-time tweet analysis.

---

## Responsible Use

This system is intended for research, educational, and decision-support use. Sentiment and sarcasm predictions are probabilistic model outputs and should not be treated as definitive statements about a person's intent, beliefs, or emotional state.

When integrating the backend with social-media products, follow the applicable platform terms, privacy requirements, and data-handling policies.

---

## License

Add the license appropriate to the final project distribution before publication.

---

## Author

**Iffat Anees Ansari**

Research-oriented NLP / AI engineering project focused on practical sentiment analysis, sarcasm-aware inference, explainability, and deployable backend systems.

---

## Status

**Backend:** Core inference + FastAPI + tests + SHAP integration complete.

**Chrome extension:** Frontend integration phase.

**Docker:** Containerization configured; fully self-contained model-artifact packaging is a deployment follow-up.

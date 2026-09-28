# Results Summary

## Overview

Three audio language models were evaluated on their ability to interpret scalar implicatures conveyed through prosodic marking. Each model completed 45 items per condition across three conditions: **marked_audio** (prosodically cued implicature), **neutral_audio** (flat delivery, literal reading expected), and **transcript_only** (text baseline). Items spanned three Gricean maxim categories: manner, quality, and quantity (15 items each).

---

## Key Findings

### 1. Implicature Detection by Condition

| Model | Marked Audio | Neutral Audio | Transcript Only |
|---|---|---|---|
| kimi-audio-7b-instruct | 6.7% | 2.2% | 2.2% |
| qwen2-audio-7b-instruct | 100% | 100% | 100% |
| step-audio-2-mini | 100% | 88.9% | 46.7% |

**Kimi** almost never inferred an implicature regardless of condition — it defaulted strongly to literal readings.

**Qwen2** always inferred an implicature regardless of condition — it defaulted to implicature even when the audio was neutral, suggesting it is driven by lexical content rather than prosody.

**Step-audio-2-mini** showed the highest implicature rate in marked audio and a meaningful drop in neutral audio, but still over-inferred implicatures relative to baseline.

### 2. Reading Accuracy (Literal vs. Implicature Classification)

In the marked_audio condition, reading accuracy reflects whether the model correctly classified the utterance as carrying an implicature; in neutral_audio, it reflects correct literal classification.

| Model | Marked Accuracy | Neutral Accuracy |
|---|---|---|
| kimi-audio-7b-instruct | 6.7% | 97.8% |
| qwen2-audio-7b-instruct | 100% | 0% |
| step-audio-2-mini | 100% | 11.1% |

Kimi correctly handles neutral audio but fails to detect prosodically marked implicatures. Qwen2 and Step-audio are the inverse: perfect on marked audio, but they almost never recover a literal reading on neutral items.

### 3. Sensitivity to Prosodic Marking (Paired Analysis)

The paired analysis compared each model's response on matched marked/neutral item pairs to assess whether prosody actually shifted the interpretation.

| Model | Pairs Showing Contrast | Rate |
|---|---|---|
| kimi-audio-7b-instruct | 2 / 45 | 4.4% |
| qwen2-audio-7b-instruct | 0 / 45 | 0% |
| step-audio-2-mini | 5 / 45 | 11.1% |

No model showed statistically significant sensitivity (exact two-sided p: Kimi = 0.50, Qwen2 = 1.0, Step = 0.0625). Step-audio-2-mini came closest but did not reach significance.

### 4. Maxim Identification Accuracy

When a model correctly identified an implicature, it was also asked to label the violated Gricean maxim. Accuracy was generally poor.

- **Kimi**: Maxim accuracy was 0% in marked_audio across all three maxim types; near-perfect in neutral_audio (where it correctly identified literal readings, making maxim labeling mostly moot).
- **Qwen2**: Achieved 100% maxim accuracy only on quality items in marked audio. Manner and quantity maxim accuracy was 0% in both audio conditions.
- **Step-audio-2-mini**: Partial accuracy — highest on quality (20% in marked, 80% in neutral), but overall still low (13% joint accuracy in marked audio).

### 5. Auditory Cue Usage

Models were asked to identify which auditory features (if any) guided their response.

- **Kimi**: Reported no auditory cue for 93% of marked_audio responses; timing was cited in rare cases. Largely ignores prosodic information.
- **Qwen2**: Consistently cited speech_rate for all responses in both audio conditions — a stereotyped response pattern not meaningfully linked to actual prosodic content.
- **Step-audio-2-mini**: Most varied; cited timing (62%), stress (24%), pitch (4%), and multiple cues (9%) in marked_audio. Even in neutral_audio, timing and stress were frequently cited, suggesting some over-attribution.

---

## Summary

None of the three models demonstrate reliable prosodic implicature detection. The models occupy three failure modes:

- **Kimi** is prosody-blind and defaults to literal interpretation.
- **Qwen2** is prosody-blind and defaults to implicature interpretation.
- **Step-audio-2-mini** shows weak sensitivity to prosody (best paired contrast rate at 11.1%, p = 0.06) but over-infers implicatures even in neutral contexts and misidentifies maxim types.

All three models achieve 100% format validity and high confidence scores (0.77–0.94), indicating that low performance is not an artifact of response formatting or model uncertainty, but reflects genuine limitations in prosodic pragmatic reasoning.

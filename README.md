# Video Anonymizer

An OCR-based pipeline for automatically detecting and redacting patient names from medical monitoring videos. Built in collaboration with [Volta Medical](https://www.volta-medical.com/) as part of a student project at École Centrale Méditerranée.

---

## Overview

Medical procedure videos — such as cardiac surgery recordings — may contain patient names displayed on monitoring screens. Sharing or archiving these videos without removing such identifiers risks violating patient privacy regulations. This tool automates the redaction process using a combination of text detection, OCR, and Named Entity Recognition (NER).

The pipeline processes MP4 videos recorded from multi-screen surgical setups (1920×1080, 25 fps), detects scenes where text changes, identifies person names using NLP, and applies pixel-level masks to obscure them — outputting a clean, anonymized MP4.

---

## Pipeline

```
Input MP4
   │
   ▼
Screen Splitting (OpenCV)       — splits video into 4 quadrant subscreens
   │
   ▼
Scene Detection (PySceneDetect) — detects visual changes to avoid per-frame OCR
   │
   ▼
Text Detection (EAST + YOLOv8)  — locates bounding boxes of text regions
   │
   ▼
Text Recognition (Tesseract)    — reads text content from detected regions
   │
   ▼
Named Entity Recognition (spaCy)— filters bounding boxes to keep only person names
   │
   ▼
Masking (OpenCV)                — blacks out identified name regions frame-by-frame
   │
   ▼
Screen Merging (OpenCV)         — reconstructs the full-resolution output video
   │
   ▼
Output MP4 (anonymized)
```

---

## Key Design Decisions

**Why scene detection?** Processing every frame at 25 fps was too slow regardless of model. Since surgical monitoring videos are largely static, scene detection triggers OCR only when the screen content meaningfully changes — reducing calls by an order of magnitude.

**Why split the screen?** Each of the 4 quadrants is an independent monitoring display. Splitting allows per-screen scene detection thresholds and prevents a change in one quadrant from masking changes in another. It also reduces image size fed to text detectors, improving detection of small text.

**Why EAST + YOLO together?** EAST excels at detecting clear, standard-sized text. YOLOv8 (fine-tuned on custom data) handles small, blurry text — typically 5–8 pixels tall — that off-the-shelf models miss. Combining their bounding boxes gives broader coverage.

**Why NER after OCR?** Text detection intentionally casts a wide net. NER (via spaCy) filters the recognized strings to keep only those classified as person names, avoiding redaction of medical values, timestamps, or device labels.

---

## Models Used

| Component | Model | Notes |
|---|---|---|
| Text detection | EAST (Frozen) | Pre-trained, good on clear text |
| Text detection | YOLOv8 XL | Fine-tuned on ~10,000 augmented medical screenshots |
| Text recognition | Tesseract OCR | Google open-source OCR engine |
| NER | spaCy `en_core_web_sm` | + `fr`, `ru`, `es` models for multilingual support |

---

## Performance

Benchmarked on a standard laptop (no GPU):

| Frames | Duration | Total time | Scenes |
|---|---|---|---|
| 170 | 6 s | 460 s | Many |
| 600 | 20 s | 135 s | 1 |
| 900 | 30 s | 146 s | 1 |
| 3000 | 120 s | 524 s | Several |

For typical 2–3 hour surgical videos with scene changes every ~10 minutes, expect roughly **2× real-time** processing (a 3-hour video takes ~6 hours). The bottleneck is text detection (EAST: ~2–3 s/frame, YOLO: ~3–5 s/frame from CLI).

---

## Limitations

- **Scrolling text**: If a name scrolls slowly across the screen without triggering a scene change, it may not be fully masked on every frame. Volta Medical confirmed this case is unlikely in practice.
- **Very small / blurry text**: Names rendered at 5–8 px height remain challenging even for the fine-tuned YOLO model.
- **Processing speed**: Does not meet real-time requirements; intended for offline overnight batch processing.
- **NER precision trade-off**: Applying NER reduces false positives but lowers precision slightly (F1 drops from 1.0 to 0.84 on the test set). The recall of 0.90 means ~10% of names may not be caught.

---

## Training Data

Due to medical data confidentiality, only 6 anonymized screenshots were available from Volta Medical. A synthetic dataset of ~10,000 images was generated for YOLO fine-tuning by:

- Placing randomly generated names at random positions on real subscreen crops
- Applying Gaussian blur and downsampling to simulate low-resolution text
- Augmenting with color, rotation, and blur variations

Split: 8000 train / 1000 val / 1000 test.

---

## Requirements

- Python 3.x
- OpenCV
- PySceneDetect
- Ultralytics (YOLOv8)
- Tesseract OCR (bundled with the application)
- spaCy + language models (`en_core_web_sm`, `fr_core_news_sm`, `ru_core_news_sm`, `es_core_news_sm`)
- Pillow, NumPy

The application is packaged as a standalone executable (PyInstaller + Gooey GUI), so end users — such as hospital staff — do not need Python installed.

---

## Application

Two GUI modes are provided via [Gooey](https://github.com/chriskiehl/Gooey):

**User mode** (for hospital staff)
- Select input video
- Set output filename (default: `output.mp4`)

**Developer mode** (for Volta Medical)
- All of the above, plus toggles to enable/disable EAST and YOLO independently, and threshold controls for each model

The app, model weights (`/models`), and a README are distributed together as a single archive. The `/models` folder must remain in the same directory as the executable.

---

## Project Context

- **Client**: [Volta Medical](https://www.volta-medical.com/) — an AI-driven cardiac electrophysiology company (FDA-cleared, CE-marked)
- **Institution**: École Centrale Méditerranée
- **Supervisors**: Anne-Laure Mealier, Thomas Boudier
- **Year**: 2024

---

## Areas for Future Work

- Fine-tune YOLO further on real (anonymized) medical footage
- Explore autoencoders for denoising/sharpening low-resolution text prior to OCR
- GPU acceleration for faster throughput
- Logging and audit trail for compliance purposes

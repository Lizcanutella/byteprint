# §5 — Robust Detection of AI-Generated Images Under Real-World Transformations

Cleaned-up transcription of `competition_info.md`, whose tables arrived flattened
(headers and cells on separate lines). Content is unchanged; only the structure
has been restored. The original file is kept as the source of record.

## 5.1 Background

Generative AI tools make it easy to create highly realistic synthetic images at
scale, creating risks for online platforms: misinformation, impersonation, fraud,
and reduced trust in digital content. Detection becomes harder after images are
compressed, cropped, reposted, or lightly edited — **so robust methods matter
more than lab-only accuracy.**

## 5.2 Problem statement

Build a prototype that distinguishes AI-generated images from authentic images
**with strong robustness under realistic post-processing and redistribution**.
The goal is not only good detection on clean data but maintained accuracy after
transformation. Solutions should present a clear technical approach, an
evaluation strategy, and thoughtful discussion of trade-offs such as robustness,
generalisation, and false positives.

> Robustness is assessed against **a subset of** the following augmentations.

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG compression | quality = 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian blur | σ = 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | scale 0.5× / 0.25×, then upscale | Thumbnail generation |
| Gaussian noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color jitter | brightness / contrast / saturation ±20% | Filter apps, auto-enhance |
| Center crop | crop 80% | Profile-picture cropping, framing |

## 5.3 Constraints & scope

| Category | Details |
|---|---|
| **In scope** | Image-level AIGC detection, robustness to common image transformations, feature engineering, model design, evaluation design, error analysis, explainability ideas |
| **Out of scope** | Full production deployment, platform-wide moderation systems, non-image modalities (video, audio) |
| **Limits** | Hackathon-scale prototype, limited compute, no access to internal production systems. Optimise for a convincing proof of concept rather than a production-grade service. **Participants must use models with <2B parameters.** |
| **Allowed assumptions** | Public or properly licensed datasets; self-created transformed test cases; reasonable deployment assumptions, as long as they are stated clearly |

## 5.4 Available resources & data

- Public or properly licensed image datasets for AIGC detection and image forensics.
- Self-created transformed samples (blur, compression, cropping, color adjustment, rescaling).
- Public documentation for relevant ML/CV libraries.
- Named datasets:
  - [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
  - [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  - [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) — use the ModelScope translation button before use

### Validation set — demonstration only

A subset of WildFake is provided so teams can demonstrate performance and track
iterative improvement. **It is a reference benchmark only, does not contribute to
the final score, and must not be used during training.**

| Split | Source | Count |
|---|---|---|
| Non-AIGC | COCO val2017 | 4,998 |
| AIGC | DALL·E Advanced | 8,843 |

## 5.5 Expected deliverables

1. **Written project description (Devpost)** — how the solution addresses the
   problem; development tools; models/APIs; libraries and frameworks; datasets
   and assets used.
2. **Public code/GitHub repository** containing:
   - Well-structured, commented code covering all components.
   - **A script that takes an image directory and outputs a confidence score per
     image — a JSON file with `image_path` and `pred` for each image**, where
     `pred` is the likelihood the image is AIGC-generated.
   - A README with: project overview; setup and installation; steps to reproduce
     results; a reflection on limitations and what you would improve with more
     time; team member contributions.
3. **Demo video** — short, end-to-end, uploaded to YouTube as public, linked from
   Devpost, free of third-party trademarks or copyrighted content.
4. **Robustness evaluation summary** — a compact table or visual comparing
   performance on clean versus transformed images.
5. **Error analysis note** — representative false positives, false negatives, and
   trade-offs.

## 5.6 Judging criteria

| Criterion | Weight | Definition |
|---|---:|---|
| Technical execution | **35%** | Strong engineering fundamentals: well-structured code, thoughtful architecture, effective use of APIs/models. The demo runs reliably; technical complexity reflects deliberate, capable decision-making. |
| Innovation & problem insight | **20%** | Originality in idea and approach; sharpness of problem understanding — how clearly the challenge is framed, why it matters, how directly the solution addresses it. |
| Impact & relevance | **20%** | Clear potential value to real users or stakeholders — meaningful reach, tangible benefit, relevance beyond the hackathon prompt. |
| Feasibility & practicality | **15%** | Realistic and buildable beyond a prototype; technically and operationally sustainable; **resource usage proportionate**; architecture holds under real-world conditions; grounded rather than speculative. |
| Presentation & communication | **10%** | Clarity of communication. *(Final event only)* the pitch tells a coherent story from problem to solution to potential, and the team answers questions with depth. |

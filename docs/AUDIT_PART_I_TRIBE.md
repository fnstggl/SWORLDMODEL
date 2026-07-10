# PART I — TRIBE v2 technical audit, causal-gap map, and keep/reject

*Primary sources read: the TRIBE v2 paper "A foundation model of vision, audition, and language for in-silico
neuroscience" (d'Ascoli et al., FAIR at Meta, 2026-03-25), the HuggingFace model card
(`facebook/tribev2`), and the paper's methods §5.1–5.10. Every claim below is from those sources.*

## I1. What TRIBE v2 actually is (primary-source audit)

A **tri-modal fMRI ENCODING model**: it predicts human brain activity (BOLD signal) from video+audio+text
stimuli. It is *not* a behavioral model, *not* a decoder, and *not* a language model.

**Pipeline (methods §5.1–5.3):**
- **Inputs**: (i) video clip, (ii) audio, (iii) transcript. As many as available (modality dropout p=0.3 at
  train time → it tolerates missing modalities).
- **Frozen feature encoders** (this is the crux for our purposes):
  - **Text → Llama-3.2-3B.** For each word, prepend the preceding 1,024 words, take intermediate-layer
    token embeddings, average → `D_text=2048`, binned to a 2 Hz grid.
  - **Audio → Wav2Vec-BERT-2.0** (`D_audio=1024`).
  - **Video → V-JEPA 2** (`D_video`, projected to 384).
  - Concatenated to `D_model = 3×384 = 1152`.
- **Trainable core**: an 8-layer, 8-head **transformer** over 100 s windows (exchanges info across time) →
  adaptive-pool 2 Hz→1 Hz → a **subject block** (subject-conditional linear layer) → **20,484 cortical
  vertices (fsaverage5) + 8,802 subcortical voxels**. A special **"unseen-subject" layer** (subject dropout
  p=0.1) gives the **group-average** response zero-shot.
- **Targets**: BOLD, MSE loss. Trained on 4 "deep" datasets (25 subj, ~450 h fMRI); tested on 4 "wide"
  datasets (695 subj). Beats an optimized deep-linear FIR baseline several-fold; recovers FFA/PPA/EBA/VWFA
  (vision), language network / Broca / STS (language), TPJ + MTG (emotional/social), zero-shot.

**Individual vs group (§2.4):** it predicts the **group-average** response better than any individual's.
Individual modeling needs **~1 h of that subject's own fMRI** to finetune (low-rank subject block, rank 128).
→ **There is no way to model a *named person's* brain without that person's fMRI.** For product use TRIBE is,
at best, a *canonical/average-human* stimulus-response representation.

**Text-path reality (model card, verbatim):** *"Text inputs are automatically converted to speech and
transcribed to obtain word-level timings."* So a headline/email is **TTS-synthesized then ASR-timed** before
Llama embeddings are taken on a 2 Hz grid. Text is also TRIBE's **weakest** modality (lowest encoding score
of the three; §2.7) — it dominates only language/prefrontal cortices.

**Compute (§5.3):** feature extraction 24 h on 128×V100; training 24 h on 1×V100. Inference cost is dominated
by the three frozen encoders (Llama-3B + Wav2Vec-BERT + V-JEPA2) + TTS/ASR for text. **Requires a GPU.**

**License (model card): `CC-BY-NC-4.0` — NON-COMMERCIAL.** Plus component licenses (Llama-3.2 community
license; V-JEPA2; Wav2Vec-BERT). **Commercial use is prohibited.**

## I2. Causal-gap map — the bridge TRIBE does *not* cross

The product needs **behavior** (click, share, reply, purchase, opinion change). TRIBE outputs **predicted
BOLD**. Every proposed use crosses arrows the model never learned:

```
stimulus ─TRIBE(trained)→ predicted BOLD per vertex ─??→ neural "salience/emotion/memory" ─??→ attention
        ─??→ preference ─??→ intention ─??→ behavior (click/share/reply)  ─observed in a real dataset
                └──────────────── NOT in TRIBE; must be LEARNED from labeled behavior ─────────────┘
```

| Arrow | Empirical support in TRIBE | What's missing |
|---|---|---|
| stimulus → BOLD | **Strong** (the paper's whole result) | limited to vision/audio/semantic; fMRI temporal res (~1 s), no ms firing |
| BOLD → "neural representation of salience/emotion/memory" | Interpretive (ICA recovers networks; TPJ/language/FFA) | a *region activation* is not a *quantity of salience*; reverse inference is a known fallacy |
| representation → attention/preference/intention | **None** | unmodeled |
| intention → behavior (click/share/reply/buy) | **None** | must be fit from a real labeled dataset (e.g. Upworthy clicks) |

**Consequence:** TRIBE can only earn a place if a **learned BOLD→behavior head** on real outcome labels beats
the **same head fed the ordinary Llama-3.2-3B embedding** (the encoder already inside TRIBE's text path). If
the plain embedding does as well, the fMRI layer added nothing — it would just be an expensive, non-commercial
re-encoding of text we already have. That is the single experiment that decides everything (I6/I7).

## I3. Text capability — the practical wrinkles for message/headline use

- Raw text is accepted, but **only via TTS→ASR→word-timings** (model card). For N candidate headlines that is
  N synthesis+transcription passes before any embedding — real per-candidate cost and a synthesis-artifact
  confound (prosody the copywriter never chose).
- No natural audio/video for a headline → 2 of 3 modalities are dropout-masked; the prediction leans on
  TRIBE's **weakest** modality.
- Static text has no intrinsic 2 Hz timeline; an imposed reading grid is an assumption the training
  distribution (timed movie/podcast transcripts) did not contain → out-of-distribution risk.

**Reproducibility check to run first (I3):** embed the same 20 headlines 5× and measure variance of the
cortical prediction; keep only contrasts (emotional−neutral, concrete−abstract, one-phrase variants) that are
**stable across repeats**. Unstable contrasts are noise, not neural signal.

## I4. Feature-extraction interface (what the adapter would expose)

Disabled-by-default adapter (`swm/experimental/tribe_adapter.py`) documents the contract for, per stimulus:
1. ordinary Llama-3.2-3B text embedding (the **control** encoder, `T3`),
2. TRIBE transformer latent (`T4`),
3. predicted cortical response, 20,484-dim (`T5`),
4. ROI/network summaries — language network, TPJ, mPFC/DMN, semantic, emotional, memory, attention regions
   (`T6`) — selected from the paper's recovered networks, **not** cherry-picked by name,
5. candidate-to-candidate neural *contrasts* (paired worlds across headline variants),
6. reproducibility variance per feature.

It caches **one TRIBE pass per unique stimulus** → many agents read the cached representation (the only
economically plausible pattern).

## I6/I7. The decision experiment — Upworthy representation ablation (design, ready to run on a GPU box)

Upworthy Research Archive (randomized headline A/B tests, real clicks; see `docs/DATASET_REGISTRY.md`, P0).
For each package's competing headlines, hidden outcomes, **article-disjoint + time-forward** splits:

| Arm | Representation fed to the SAME small ranking head |
|---|---|
| T0 | raw stimulus id only (no features) |
| T1 | grounded DeepSeek one-shot pick |
| T2 | ordinary sentence embedding (non-Llama, e.g. MiniLM) |
| **T3** | **Llama-3.2-3B embedding (TRIBE's own text encoder, no fMRI)** ← the control that TRIBE must beat |
| T4 | TRIBE transformer latent |
| T5 | TRIBE predicted cortical response |
| T6 | TRIBE low-dim network summaries |
| T7–T10 | agent panel ± TRIBE latent / summaries |

**Metrics:** precision@1, pairwise ranking accuracy, expected regret, observed click lift, calibration,
paired bootstrap significance. **Decision rule (hard rule #13):** claim neural value **only if T4/T5/T6 beat
T3** (the plain Llama embedding) on held-out clicks with a significant paired margin.

## I11. Keep / reject — **RESEARCH-ONLY (leaning reject for production)**

| Criterion | Finding |
|---|---|
| What it genuinely predicts | group-average BOLD to audio/video/semantic stimuli — strongly |
| What it does **not** predict | attention, preference, intention, **behavior** — the product's actual targets |
| Behavioral lift shown? | **None yet** — untested; must beat the plain Llama-3.2-3B embedding to count |
| Named-person modeling | **No** without that person's own fMRI (~1 h) |
| Text path | TTS→ASR per message; text is its weakest modality; OOD for static copy |
| Compute | GPU-only; 3 heavy frozen encoders + TTS/ASR per stimulus |
| **License** | **CC-BY-NC-4.0 — NON-COMMERCIAL.** Hard blocker for a commercial product. |
| Runnable in this environment? | **No** — no GPU; large HF weight pulls are proxy-limited here |

**Decision: RESEARCH-ONLY. Do not wire into production.** Two independent blockers stand before any product
use: (1) the **non-commercial license** (plus Llama/V-JEPA component terms), and (2) the **unproven
BOLD→behavior bridge** that, on priors, may not beat the ordinary Llama embedding it's built on — precisely
because text is TRIBE's weakest modality. The correct next step is **not** integration; it is the I7 Upworthy
representation ablation on a GPU box, with the plain-embedding control (T3) as the bar. Ship the adapter
**disabled and quarantined** (`swm/experimental/`, never imported by the front door) so nothing can regress a
commercial, unvalidated neural layer into the engine. Revisit only if (a) a commercial license path appears
**and** (b) T4/T5/T6 beat T3 on held-out behavior.

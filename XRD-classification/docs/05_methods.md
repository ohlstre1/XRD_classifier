# Evaluation: Methods

**Source file:** `sections/methods.tex`
**Rubric category:** Methods
**Estimated grade:** 3–4 / 5

---

## Strengths

- The XRD generation algorithm (Section 5.1) is clearly derived with proper equations showing how synthetic patterns are built from CIF files.
- The clustering analysis (Section 5.2) is a novel contribution with well-justified methodology for quantifying structural ambiguity in XRD.
- The ResNet-18 baseline (Section 5.3) is cleanly structured with clear input/output specifications.
- The diffusion model section (Section 5.4) is detailed, covering augmentation strategy, training procedure, and inference.

---

## Issues to Fix

### 1. Banned words

- **Line 135:** uses "multitude" — replace with a specific term or just state the number.
- Check the full file for other violations (e.g., references to "crucial" from other sections).

### 2. Section 5.1 is partially redundant with background/literature review

The structure factor derivation (lines 26–47) overlaps with the Rietveld equations already presented in the literature review. Reference the earlier derivation instead of repeating it, or (better) move the Rietveld equations from the literature review to here.

### 3. Normalization discussion is over-explained (lines 111–131)

The rationale for choosing min-max normalization over L1/L2 is fine, but it takes 20 lines for a straightforward decision. Compress to 5–6 lines: state the options, state the choice, give the reason.

### 4. ResNet-18 justification is weak (line 214)

Current text: "The main reason for choosing this model is because it is a famous deep learning algorithm which is similar to our task."

This is not a scientific justification. Replace with specific reasons:
- Residual connections handle vanishing gradients for this signal length.
- It is a well-understood baseline, making results reproducible and comparable.
- Its parameter count (~11M) is appropriate for the dataset size (~13K samples).

### 5. Broken reference at line 113

`\ref{eq:}` — the label is empty. This will render as "??" in the compiled PDF. Fix the label reference.

### 6. Typo at line 214

"beacuse" should be "because".

### 7. Commented-out sections (lines 403–482, 487–592)

These contain old material about AutoXRD and DTW that is no longer part of the thesis. Remove all commented-out blocks before submission.

### 8. Missing hyperparameter values for diffusion model

The augmentation equations are well-presented, but the actual probability and range values ($p_{\text{shift}}$, $p_{\text{var}}$, $p_{\text{remove}}$, $L_{\text{min}}$, $L_{\text{max}}$) are never specified. These are needed for reproducibility. Add a table or inline values.

### 9. Missing ablation justification

Why a 1D U-Net with 237K parameters specifically? Was any architecture search performed? How does this compare to simpler (e.g., MLP-based) or larger (e.g., 1M+ parameter) alternatives? Even a brief justification would strengthen the section.

### 10. Summary paragraph is too long and list-like (lines 133–143)

This reads as if listing every possible limitation of synthetic XRD. Pick the top 3 that your diffusion model actually addresses and focus on those.

---

## Path to Grade 4–5

1. Fix the broken `\ref{eq:}` reference and the "beacuse" typo.
2. Remove all commented-out sections.
3. Eliminate redundancy with the background/literature review chapters.
4. Justify the ResNet-18 choice with specific technical reasons.
5. Specify all diffusion model hyperparameters for reproducibility.
6. Add brief justification for the U-Net architecture choice.
7. Compress the normalization and summary discussions.

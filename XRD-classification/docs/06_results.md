# Evaluation: Results

**Source file:** `sections/results.tex`
**Rubric category:** Results and Contribution
**Estimated grade:** 3–4 / 5

---

## Strengths

- Three distinct result blocks (clustering, ResNet baseline, diffusion) each with supporting tables and figures.
- The clustering analysis with sensitivity table and silhouette scores is thorough and methodical.
- Diffusion model results include stochastic variation analysis, noise-level waterfall plots, and quantitative metrics.
- Figures are well-chosen and illustrative of the key findings.

---

## Issues to Fix

### 1. Text vs. figure inconsistency (around line 110)

The body text reports "distance of 0.129" and "distance of 0.291", but the corresponding figure captions say "Distance 0.104" and "Distance 0.385". These numbers must match. Determine which set is correct and fix the other.

### 2. Stray period and whitespace (line 83)

".  " — a random period with trailing whitespace. Remove.

### 3. Clustering results narrate the table

Lines 149–163 essentially read Table 3 row by row: "at threshold 0.20 we get X, at 0.15 we get Y..." This is redundant with the table itself. Instead, state the key finding (metrics improve monotonically with tighter thresholds) and highlight 2–3 specific points of interest or surprising results.

### 4. ResNet results are thin

Only one table with three numbers (76.70% top-1, 93.30% top-3, 95.20% top-5). This needs:
- **Per-class accuracy distribution:** A histogram or confusion matrix excerpt showing which crystal systems or phases the model handles well vs. poorly.
- **Error analysis:** What types of phases does the model confuse? Are the confusions consistent with the clustering analysis (i.e., phases that share near-identical XRD patterns)?
- **Connection to clustering ceiling:** The 3.6–6.7% theoretical overlap from clustering should set an upper bound on avoidable errors. Compare this to the observed error rate.

### 5. Identical Pearson correlation and NCC values

The diffusion model results report Pearson correlation and normalized cross-correlation as identical (0.872 $\pm$ 0.185). If this is correct, explain why these two different metrics converge (e.g., because the data is zero-mean after normalization). If it is a copy-paste error, fix it.

### 6. Missing pre-diffusion vs. post-diffusion classification comparison

The thesis claims a pipeline (diffusion $\to$ classification) but never shows whether the diffusion step actually improves classification accuracy. This is a major gap. Add a comparison:

| Training data | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Raw synthetic | X% | X% | X% |
| Diffusion-augmented | Y% | Y% | Y% |

Without this, the pipeline claim is unsubstantiated.

### 7. Missing connection between clustering and classification

The clustering shows 3.6–6.7% overlap, which implies a theoretical accuracy ceiling of ~93–96%. But the ResNet achieves only 76.7% top-1, meaning 23.3% error rate — far exceeding the theoretical minimum. This discrepancy deserves analysis. Is the gap due to insufficient training data, the domain gap between synthetic and real patterns, or model capacity?

### 8. The 93% ICDD round-robin accuracy claim

Discussion section (line 53) mentions "93% top-1 accuracy on the ICDD round-robin set", but this result never appears in the results chapter. If this experiment was conducted, report it here with full methodology. If it has not been done, remove the claim from the discussion.

---

## Path to Grade 4–5

1. Add the diffusion-vs-no-diffusion classification comparison — this is the single most important addition.
2. Connect the clustering ceiling to classification accuracy with explicit analysis.
3. Fix the text/figure numerical inconsistencies.
4. Add error analysis for ResNet (confusion patterns, per-class accuracy).
5. Clarify or fix the identical Pearson/NCC values.
6. Report the ICDD round-robin experiment here, or remove the claim from the discussion.

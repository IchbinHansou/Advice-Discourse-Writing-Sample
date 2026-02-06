# Gold Labeling & Evaluation Notes v1

This file does not duplicate the labeling rules.
Canonical intent-label definitions are in: `docs/annotation_guidelines.md`.

## 1) What “Gold” means in this project
- “Gold” refers to the manually annotated subset used for evaluation and analysis.
- The gold label is still **intent-based** (same definitions as the canonical guideline).
- Domains (`ra`, `confession`, `aita`) remain domains only.

## 2) Binary evaluation policy (ADVICE vs STORY)
Primary headline metrics in the paper are computed on the **binary task**:

- Keep only `ADVICE` and `STORY` for macro-F1.
- `MIXED` is excluded from the primary binary macro-F1 and is analyzed separately.

If you maintain a derived file like `gold_binary.csv`, it should contain only:
- `doc_id`, `domain`, `text_*`, and `gold_label` in {`ADVICE`,`STORY`}.

## 3) When you suspect label noise (single-annotator reality)
If you see a cluster of “errors” (e.g., AITA misclassifications), treat it as a hypothesis:
- could be model error,
- could be label ambiguity,
- could be your own labeling inconsistency.

Recommended audit workflow:
1) Extract error cases for a domain (e.g., AITA) into a CSV.
2) Add an `audit_label` column and re-label from scratch using the canonical guideline.
3) Record a short `category` (typology) only to explain *why* it was hard (optional).
4) Update the official `gold_label` only when you are confident the original was wrong.

Practical rule for updates:
- keep the old label in a separate column (e.g., `gold_label_v0`) before overwriting,
- log changes in a short changelog entry (date + how many changed + why).

## 4) AITA-specific reminder (v1 policy)
In AITA, “verdict / evaluation request” counts as `ADVICE` when it is the central goal
(e.g., “am I the asshole?”, “was I wrong?”, “judge my conduct?”).
Do not demote these to `STORY` only because the post reads like a narrative; the narrative is often the context for the verdict request.

A common borderline pattern:
- “long story + one final ‘AITA?’” can still be `ADVICE` if the entire post is framed to obtain a judgment.

## 5) Self-consistency check (recommended)
If time allows, re-label a random 10% subset after ~7 days and report:
- percent agreement (and optionally Cohen’s kappa).
State clearly if you are a single annotator and no second annotator is available.

## 6) Versioning
- v1 = rules in `docs/annotation_guidelines.md` at the time you freeze your writing sample experiments.
- If you change the policy later (e.g., adjust how you treat specific borderline AITA patterns), create v2 and summarize changes in a small `docs/guideline_changelog.md`.
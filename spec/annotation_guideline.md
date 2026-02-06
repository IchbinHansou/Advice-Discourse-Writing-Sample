# Annotation Guidelines v1 (Intent Labels)

## 1) Scope and Unit
This project labels the **pragmatic intent** of a Reddit **submission**.

- **Unit:** one submission (title + body/selftext).
- **Ignore comments** (not provided; not annotated).
- **Domains are not labels:** `ra`, `confession`, `aita` are treated as **domains** only. The label is decided by the post’s **communicative goal**, not the subreddit name.

## 2) Label Set (3-way)
Assign exactly one label:

- `ADVICE`
- `STORY`
- `MIXED`

## 3) Core Principle (Intent-Centered)
Pick the label that best matches the post’s **central communicative goal**:

- `ADVICE` when the post primarily **solicits guidance / judgment / decision support**.
- `STORY` when the post primarily **discloses a narrative / vents / shares feelings**, and actionable guidance is not the main point.
- `MIXED` when **both** advice-seeking and narrative disclosure are **central and substantial**.

A simple mental check:
If the “best possible reply” would mainly be **instructions / recommendations / next steps**, push toward `ADVICE`.
If the “best possible reply” would mainly be **empathy / listening / shared experience**, push toward `STORY`.

## 4) Operational Definitions

### 4.1 ADVICE (Guidance / decision support is central)
Label `ADVICE` when the post’s main point is to get input on **what to do**, **how to handle a situation**, or **whether a decision/behavior is right**.

Common cues (any subset is enough if central):
- explicit solicitation: “What should I do?”, “How do I…?”, “Any advice?”, “Should I…?”
- decision framing: choosing between options, asking for steps or a plan
- evaluative / verdict request: “Was I wrong?”, “Am I being unreasonable?”, “Am I the asshole?”

The post can contain lots of background story. It still counts as `ADVICE` if the narrative mainly functions as context for the request.

#### 4.1.x Implicit advice-seeking (no explicit question)
Some posts don’t include a clean “What should I do?” line, but the intent is still guidance-seeking.

Label `ADVICE` if the post:
- presents a **current dilemma / decision point** (with constraints or trade-offs),
- signals being stuck (“I don’t know what to do”, “I’m at a loss”), and
- would mostly receive actionable **recommendations / next steps** as the best reply.

Do **not** require a question mark or a formulaic advice phrase.

Counter-check:
If the post mainly recounts events/feelings and the implied response is primarily empathy/validation, prefer `STORY`.
If both advice-seeking and narrative disclosure are central, prefer `MIXED` (see §4.3).

### 4.2 STORY (Narrative disclosure / affective sharing is central)
Label `STORY` when the post’s main point is telling what happened / how they feel, without a clear or central request for guidance.

Common cues:
- self-contained account: events + feelings + reflection
- questions are rhetorical / minor, or do not truly ask for action-oriented guidance
- the implied response is empathy (“that sounds hard”), not “do X next”

A post may contain question marks and still be `STORY` if it does not meaningfully ask the reader to provide guidance.

### 4.3 MIXED (Both intents are central and substantial)
Use `MIXED` only when:
1) a substantial portion is narrative disclosure (not just a few lines of context), **and**
2) a substantial portion is advice/judgment request that is also central (not a throwaway line).

Guardrail (to avoid overusing `MIXED`):
- If the advice request is a single line appended to a long vent (“any advice?”) but the post still makes sense as emotional disclosure without it, prefer `STORY`.
- If the narrative is mostly context and the post would feel incomplete without the request, prefer `ADVICE`.

## 5) Domain-Specific Notes (AITA is the hard domain)
Many AITA submissions contain boilerplate artifacts (e.g., “AITA?”, “WIBTA”, “YTA/NTA/ESH/NAH/INFO”).

- Do not label based on artifact tokens alone.
- If the core goal is “judge my conduct / tell me if I’m wrong”, treat this as evaluative decision support → `ADVICE`.
- If the post is mainly an account/vent and any verdict phrase looks like boilerplate, prefer `STORY`.
- If there is substantial disclosure plus a substantial “what should I do next / how do I handle this” beyond verdict-seeking, use `MIXED`.

## 6) Artifact-Ignoring Rule (applies to all domains)
This project compares Raw vs Cleaned text (cleaning removes community markers). During annotation, do not rely on templated artifacts that can leak domain identity, such as:
- section markers: `AITA`, `WIBTA`, `TL;DR`, `Edit:`, `Update:`
- AITA verdict abbreviations: `YTA`, `NTA`, `ESH`, `NAH`, `INFO`
- template metadata: `[25M]`, `(30F)`, “throwaway”, “x-post”

If the post’s only “advice signal” is an artifact token, treat that as insufficient and label based on the remaining content.

## 7) Practical Decision Procedure (fast, consistent)
1) Read title + body once for gist.
2) Identify whether there is an explicit or implicit request for guidance/judgment.
3) Decide what is central: guidance-seeking vs disclosure.
4) Apply guardrails:
   - throwaway request appended to a long vent → tends toward `STORY`
   - story mostly context for a decision → tends toward `ADVICE`
   - both clearly central → `MIXED`

## 8) Notes Field (optional)
Use notes only for borderline cases (one short line), e.g.:
- “Mostly venting; advice line is minor → STORY”
- “Verdict-seeking is central despite narrative → ADVICE”
- “Both explicit request + substantial disclosure → MIXED”

## 9) Label Vocabulary (exact strings)
Use exactly: `ADVICE`, `STORY`, `MIXED`
No lowercase, no extra labels.
# Gold Annotation Guideline v1

## 1) Scope and Unit
- **Unit:** one Reddit submission (title + body/selftext).  
- **Ignore comments** (not provided, not annotated).
- **Domains are not labels:** `ra`, `confession`, `aita` are treated as *domains* only. The gold label must be based on the post’s **pragmatic intent**, not its subreddit.

## 2) Label Set
Use exactly one of:
- `ADVICE`
- `STORY`
- `MIXED`

`MIXED` is **excluded** from the primary ADVICE–STORY macro-F1 and is analyzed separately.

## 3) Core Principle (Intent-Centered)
Assign the label that best matches the post’s **central communicative goal**:
- If the post primarily **solicits guidance / judgment / decision support**, label `ADVICE`.
- If the post primarily **discloses a narrative / vents / seeks empathy or validation without needing actionable guidance**, label `STORY`.
- If **both** advice-seeking and narrative disclosure are **central and substantial**, label `MIXED`.

A useful check: imagine the post receives replies.  
- If the “best possible reply” is mainly **instructions / recommendations / next steps**, that pushes toward `ADVICE`.  
- If the “best possible reply” is mainly **empathy / listening / shared experience**, that pushes toward `STORY`.

## 4) Operational Definitions

### 4.1 `ADVICE` (Guidance / decision support is central)
Label `ADVICE` when the post’s main point is to **get input on what to do, how to handle a situation, or whether a decision/behavior is right**.

Common cues (any subset is enough if the intent is central):
- Explicit solicitation: “What should I do?”, “How do I…?”, “Any advice?”, “Should I…?”
- Decision framing: choosing between options, asking for a plan, asking for steps to take
- Moral/behavioral evaluation sought as feedback: “Was I wrong?”, “Am I being unreasonable?”, “Am I the asshole?”

**Important:** The post can contain background story; it is still `ADVICE` if the narrative mainly functions as context for the request.

#### 4.1.x Add-on Rule: Implicit Advice-Seeking (no explicit question)

Some posts do not contain an explicit question (e.g., “What should I do?”), but still have a **central guidance-seeking intent**.

Label `ADVICE` if the post:
- clearly presents a **current dilemma / decision point** (often with constraints, trade-offs, or jealousy/confusion),
- signals inability to proceed (“I don’t know what to do”, “I’m stuck”, “I’m at a loss”), and
- would receive the “best possible reply” mainly as **actionable recommendations / next steps**, rather than empathy-only.

Do **not** require a question mark or a formulaic advice phrase. The decision-seeking intent can be implicit.

Counter-check:
- If the post mainly recounts events/feelings and the implied response is primarily empathy/validation (even if it ends with a minor or rhetorical question), prefer `STORY`.
- If substantial disclosure and a central guidance goal are both present, prefer `MIXED` per §4.3 and its guardrails.

### 4.2 `STORY` (Narrative disclosure / affective sharing is central)
Label `STORY` when the post’s main point is **telling what happened / how they feel**, without a clear or central request for guidance.

Common cues:
- The post reads like a self-contained account: events + feelings + reflection
- Questions are rhetorical, minor, or not requesting actionable guidance
- The implied response is empathy (“that sounds hard”), not “do X next”

**Important:** A post may contain a question mark and still be `STORY` if it does not meaningfully ask the reader to provide guidance.

### 4.3 `MIXED` (Both intents are central and substantial)
Use `MIXED` only when:
1) A substantial portion is narrative disclosure (not just a few lines of context), **and**
2) There is an explicit advice/judgment request that is also central (not a throwaway line).

Typical structure:
- Background → conflict/emotion → explicit request for guidance/judgment (and the request is a major part of the post’s purpose)

**Guardrail (to avoid overusing MIXED):**
- If the advice request is a single line tagged onto a long vent (“any advice?”) but the post would still make sense as emotional disclosure without it, prefer `STORY`.
- If the narrative is mostly context and the post would feel incomplete without the request, prefer `ADVICE`.

## 5) AITA-Specific Guidance (Challenge Domain)
Many AITA submissions include community artifacts (e.g., “AITA?”, “WIBTA”, “YTA/NTA/ESH/NAH/INFO”). For gold labeling:

- **Do not label based on artifact tokens alone.** Label based on the underlying intent:
  - If the post’s core goal is “tell me whether I was wrong / judge my conduct,” treat this as **decision support / evaluative feedback** → label `ADVICE`.
  - If the post is mainly an account/vent with no real request for judgment beyond boilerplate, label `STORY`.
  - If the post contains both a long disclosure *and* a central, explicit “what should I do now / how do I handle this” beyond verdict-seeking, label `MIXED`.

## 6) Artifact Ignoring Rule (applies to all domains)
Your project evaluates Raw vs Cleaned text, where the cleaned condition removes community markers. During annotation, **do not rely on** templated artifacts that can leak domain identity, such as:
- Explicit tags and section markers: `AITA`, `WIBTA`, `TL;DR`, `Edit:`, `Update:`
- AITA verdict abbreviations: `YTA`, `NTA`, `ESH`, `NAH`, `INFO`
- Template metadata: age/gender brackets like `[25M]`, `(30F)`, “throwaway”, “x-post”

If the post’s only “advice signal” is an artifact token, treat it as insufficient and label based on the remaining content.

## 7) Practical Decision Procedure (fast, consistent)
When labeling a post, do this in order:
1) Read title + body once for gist.
2) Identify whether there is an explicit or implicit request for guidance/judgment (one sentence is enough).
3) Decide what is **central**: guidance-seeking vs disclosure.
4) Apply guardrails:
   - Minor/throwaway request appended to a long vent → tends toward `STORY`
   - Story is mostly context for a decision → tends toward `ADVICE`
   - Both are clearly central → `MIXED`

## 8) Notes Field (when to use it)
Fill `notes` only for borderline cases. Keep it short (one line):
- “Mostly venting; advice line is minor → STORY”
- “Verdict-seeking is central despite narrative → ADVICE”
- “Both explicit request + substantial disclosure → MIXED”

## 9) Quality Control (single-annotator)
- After ≥ 7 days, re-label a random **10%** subset of gold posts (same label set).
- Report intra-annotator agreement (percent agreement; optionally Cohen’s kappa).
- If constraints prevent a second annotator, state that clearly in the paper.

## 10) Label Vocabulary (exact strings)
Use exactly: `ADVICE`, `STORY`, `MIXED`
No lowercase, no extra labels.
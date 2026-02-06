## Table 1 — Gold-set label distribution by domain (FROZEN)
| domain     |   ADVICE |   STORY |   MIXED |   Total |
|:-----------|---------:|--------:|--------:|--------:|
| aita       |       28 |      64 |       8 |     100 |
| confession |       30 |      58 |      12 |     100 |
| ra         |       89 |       9 |       2 |     100 |
| Total      |      147 |     131 |      22 |     300 |

## Table 2 — Length summary (medians; whitespace-delimited words) (FROZEN)
| Split / Label        |    N |   Word median |
|:---------------------|-----:|--------------:|
| Silver (weak, train) | 1630 |           226 |
| Gold (manual, eval)  |  300 |           319 |
| Gold — Advice        |  147 |           312 |
| Gold — Disclosure    |  131 |           317 |
| Gold — Mixed         |   22 |           446 |

As detailed in Table 1, the manual annotation yielded 147 Advice and 131 Disclosure posts, alongside 22 Mixed-intent posts (7.33% of the gold set). For the primary binary evaluation, we exclude Mixed posts, resulting in a final evaluation subset of 278 posts.

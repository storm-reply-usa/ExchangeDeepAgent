---
name: severity
description: Assign a severity from the first numbers. Use at assess, before anything else, because everything downstream follows from it.
---

> **How grave the agent thinks this is.** It applies this first, and everything
> after follows from the answer. Change the thresholds and it calls the same
> night something else.

Assign a severity of 1 to 4 before you do anything else.

Weigh three things, in this order:

1. How many customers are affected.
2. What they cannot do.
3. Whether money is involved. Money taken and not confirmed is worse than
   money not taken.

| Sev 1 | nobody can buy, or money is at risk |
| Sev 2 | many customers cannot finish something important |
| Sev 3 | a feature is broken and there is a workaround |
| Sev 4 | visible, not costly |

Report the level, then the figures behind it. If a figure is an estimate, say
so rather than presenting it as measured.

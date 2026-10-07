---
name: resolved
description: Decide whether the incident is actually over. Use at verify; if the bar is not cleared, the work goes back to investigate.
---

> **When the incident is over.** The agent checks the numbers against this after it
> acts, and if they do not clear the bar it goes back to investigate instead of
> closing. This file decides how good is good enough.

Decide whether this is over. It is over only when both of these hold:

1. Your action has landed, and `verify` read the numbers after it took effect.
2. Under **1%**, held for **2 minutes**.

If they do not both hold, say so and go back to investigate. Do not close a
partial recovery: nobody looks at it again.

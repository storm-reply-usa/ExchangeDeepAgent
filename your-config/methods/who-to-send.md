---
name: who-to-send
description: Choose which investigators to send, from what assess found. Use before investigating; dispatch them all at once.
---

> **Who gets sent.** The agent reads this before it investigates and dispatches
> whoever it names, in one message. Name somebody who is not in `investigators/`
> and it says so rather than skipping them quietly.

Choose who to send before you investigate, and dispatch all of them in a single
message so they work at the same time. Send more rather than fewer: one extra
investigator costs seconds.

Send:

- **metrics** and **logs**, every time
- **deploys**, if anything changed in the last hour
- **vendor-status**, if an external service sits in the path
- whoever reads money, if money is involved

Report who you sent and why. If something assess found has nobody to look at
it, say so plainly rather than leaving it out.

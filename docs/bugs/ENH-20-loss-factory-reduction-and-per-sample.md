Title: Allow configuring loss reduction and expose per-sample losses

Severity: low
Status: open

Subsystem: agent/loss
Affected Version/Branch: main

Description:
- LossFactory returns reduced loss modules only; PER path needs per-sample losses (reduction='none'), and some users may prefer 'sum' over 'mean'.

Proposed Enhancement:
- Add `reduction: Literal['mean','sum','none']` to LossConfig; return a module or function accordingly.
- Provide a helper to get per-sample losses for PER even when using global loss config.

Migration Impact:
- Backwards compatible default 'mean'. PER code can opt into 'none'.

Tests:
- Verify behavior for each reduction option.

Owner: agent

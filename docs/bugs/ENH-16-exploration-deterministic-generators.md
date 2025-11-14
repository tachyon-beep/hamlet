Title: Accept torch.Generator for deterministic action selection and RND sampling

Severity: low
Status: open

Subsystem: exploration
Affected Version/Branch: main

Description:
- For reproducible experiments and tests, allow callers to pass a `torch.Generator` to `epsilon_greedy_action_selection` and RND update paths.

Proposed Enhancement:
- Add optional `generator: torch.Generator | None` to `epsilon_greedy_action_selection` and use it in `multinomial`/`rand`/`randint`.
- Thread the generator through exploration strategy APIs if needed.

Migration Impact:
- Backwards compatible; default randomness unchanged.

Tests:
- Verify identical actions for fixed seeds/generators.

Owner: exploration

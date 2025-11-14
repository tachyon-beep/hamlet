Title: SequentialReplayBuffer samples episodes uniformly, not by transitions

Severity: low
Status: open

Subsystem: training/replay-buffer (sequential)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/sequential_replay_buffer.py:146`

Description:
- Sampling uses `random.choice(valid_episodes)` (uniform over episodes), then uniform start index.
- Longer episodes contribute more valid start positions, but are underweighted by episode-uniform selection.

Reproduction:
- Track distribution of sampled start positions across episodes with varying lengths.

Expected Behavior:
- Probability proportional to number of valid start positions per episode (i.e., by transitions).

Actual Behavior:
- Uniform over episodes regardless of length.

Root Cause:
- Simplicity of `random.choice` over the episode list.

Proposed Fix (Breaking OK):
- Sample episode index with weights `max(1, len(ep) - seq_len + 1)` to reflect valid starts.

Migration Impact:
- None; only sampling distribution changes (better statistical efficiency).

Alternatives Considered:
- Keep uniform; simpler but biases learning toward shorter episodes.

Tests:
- Statistical smoke test: longer episodes should be selected more often over many samples.

Owner: training
Links:
- N/A

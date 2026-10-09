---
name: did-experiment
description: Test a research hypothesis of the DID Hack agent offline - many seeded scenarios in parallel, paired comparison of two agent variants, sign test, Markdown table for docs/EXPERIMENTS.md. Use for "does X help?" questions instead of judging by a single run.
---

# Offline experiment for a hypothesis

One Gazebo run is an anecdote. The offline simulator uses the same judge, map, A* and planner, runs a mission in seconds,
and the batch script compares variants on the same scenarios (pairs by seed).

1. State the hypothesis as "if ..., then <metric> is better than <baseline>" and pick variants:
   - H1 terrain learning: `--levels easy medium hard --seeds 30 --battery 60 25` → variants B (learns) vs A (shortest path)
   - H2 knowledge base between missions: `--knowledge --seeds 30 --battery 25` → K vs B
   - H3 LLM advisor (real model, key in `.env`, ~9 s per call): `--llm --levels medium hard --seeds 8 --battery 25 --jobs 8` → C vs B
   A new variant = a new flag in `scripts/science_batch.py` (keep `variant` in each row).
2. Run from `src/did_agent`:
   ```bash
   python3 ../../scripts/science_batch.py <args>          # prints the Markdown table, saves runs/science_batch_<time>.json
   python3 ../../scripts/sign_test.py ../../runs/science_batch_<time>.json B A --battery 25
   ```
3. Before trusting numbers: every mission returned (`returned`), no tracebacks; look at outliers with
   `python3 ../../scripts/science_debug.py <level> <seed> --battery 25 [--knowledge]` (full journal of one run).
4. Report honestly: effect size per level, wins/losses/ties, p-value; "not significant" is a valid result —
   explain it from the journals and propose the next hypothesis. Update `docs/EXPERIMENTS.md` and DEVLOG.md.

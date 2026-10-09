---
name: did-safe-commit
description: Commit a finished step of the DID Hack project the team's way - tests green, no API key in tracked files, DEVLOG entry, commit message from a file (Cyrillic/quotes safe). Use after every working step.
---

# Safe commit of a step

Team rules: a commit after every working step, a DEVLOG entry, and the LLM key never leaves `.env`.

1. Tests: `cd src/did_agent && python3 -m pytest -q` — all green (the live LLM test is skipped unless `RUN_LLM_LIVE=1`).
2. Secrets: the key lives only in `~/innopolis_proj/.env` (gitignored, mode 600). Scan the tree; any hit stops the commit:
   ```bash
   grep -rIl "sk-" --exclude-dir=.git --exclude=.env --exclude-dir=build --exclude-dir=install . 
   ```
   Also never print the key in logs, prompts saved to `runs/` or screenshots.
3. DEVLOG.md: append "## <date> — Шаг N: <title>" with what changed, why, the proof (numbers, logs) and the pitfalls ("Грабли").
   QUESTIONS.md: a new ambiguity of TASK.md → a row with the temporary decision.
4. Commit with a message file — heredocs with quotes and Cyrillic broke `git commit -m` before:
   ```bash
   cat > .git/COMMIT_MSG_TMP <<'EOF'
   <summary line>

   <body>
   EOF
   git add <paths> && git commit -q -F .git/COMMIT_MSG_TMP
   ```
   Stage paths explicitly (never `runs/`, `build/`, `install/`, `.env`).
5. Push only when the user asked for it in this session or earlier for this repository.

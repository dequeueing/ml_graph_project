# GitHub export scope

This repository preserves the accepted results of `scoring-v1-20261122`. The original local project retains its complete, unmodified frozen snapshot.

One saved GLM answer, `topology/88/dfs/zero_shot_cot/r1`, contains a credential-like string appended after the graph answer. Its validity as a credential was not tested. The export replaces that string with `[REDACTED_CREDENTIAL_LIKE_STRING]` in `analysis/scoring_audit_inputs.json.gz`. The original file is omitted from `analysis/releases/scoring-v1-20261122/` so that it cannot be uploaded inadvertently.

`PUBLICATION_MANIFEST.json` records the model, case ID, original file hash, exported file hash, and omitted path. All other 39 files listed in the acceptance manifest are included unchanged. The original acceptance manifest is also unchanged, so its input-file hash deliberately refers to the locally retained original rather than the redacted export.

No scores, review decisions, graph metadata, or prompts were changed. This export did not rerun scoring or establish a new accepted scoring version. The source-hash fields in existing results retain their original meaning. Exact verification of the omitted original and exact replay of the original acceptance require access to that original input in the local project. The portable verification script checks the included files, the redacted export's separate hash, experiment prompt alignment, and current report images.

The original SQLite run databases remain on the experiment server. Credentials, installed environments, downloaded dataset caches, unrelated course materials, and Overleaf session files are outside this GitHub export.

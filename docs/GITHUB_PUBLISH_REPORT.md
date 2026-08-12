# Private GitHub publication audit

Publication completed and verified on 2026-08-12.

## Repository

1. Repository name: `PaFAR`
2. Owner: `gaozw23`
3. Visibility: `PRIVATE` (verified from the GitHub API via GitHub CLI)
4. Remote URL: `https://github.com/gaozw23/PaFAR`
5. Branch: `main`; GitHub default branch: `main`
6. Initial commit: `76c90e24467f7008a8b88d59fd7cf6eb990842eb` (`Initial private research code and manuscript`)
7. Final tracked file count after this report: 200
8. Final tracked working-tree size after this report: 4,566,958 bytes (4.355 MiB). The pre-report upload candidate was 199 files and 4,563,413 bytes (4.352 MiB).

## Exclusions and safety checks

9. Excluded directories and categories:
   - `.venv/`, Python caches, and editor-local files;
   - `data/**` except `data/README.md`;
   - `outputs/**` except `outputs/README.md`;
   - `archives/`, `tmp/`, failure caches, logs, and compressed archives;
   - memmaps, NumPy arrays, serialized models, checkpoints, and bootstrap objects;
   - credentials, `.env` files, private-key formats, and downloaded third-party reference material;
   - LaTeX build products, manuscript backups, and superseded rendered drafts.
10. Test result: passed, `102 passed in 41.28s`, using `.venv/Scripts/python.exe -m pytest -q`. No simulation or real-data analysis was rerun.
11. Secret scan: passed. Candidate text files had zero matches for the configured token prefixes, private-key headers, credential assignments, or HTTP credential markers. Reports did not print secret values.
12. Large-file scan: passed. No candidate file exceeded 10 MiB or 50 MiB; the largest candidate was 775,612 bytes.
13. Remote forbidden-path audit: passed. `origin/main` initially contained 199 tracked files and zero forbidden raw-data, cache, model, checkpoint, archive, secret, or third-party-reference paths.
14. Clone verification: passed. A private depth-1 clone used branch `main`, contained all required repository paths, contained none of the checked forbidden paths, and had a clean working tree. The temporary clone was removed afterward.
15. Raw PhysioNet data uploaded: no. `data/physionet2019/raw/` is ignored and absent from the remote tree.
16. Feature cache uploaded: no. `data/physionet2019/cache/`, memmaps, NumPy arrays, and serialized feature objects are ignored and absent from the remote tree.
17. Simulation raw checkpoints uploaded: no. `outputs/production/`, raw output files, checkpoints, and serialized objects are ignored and absent from the remote tree.
18. Real-data raw checkpoints uploaded: no. `outputs/realdata/`, fitted models, checkpoints, bootstrap objects, and failure caches are ignored and absent from the remote tree.
19. GitHub authentication: GitHub CLI browser authorization for account `gaozw23`; credentials are stored in the Windows keyring, and Git operations use HTTPS. No explicit token was written to the project or report.

## Remaining recommended actions

20. Keep the repository private; review collaborator access before inviting anyone; consider a protected `main` ruleset if multiple collaborators will push; rerun the secret, large-file, and forbidden-path audits before future releases; continue obtaining PhysioNet data and the official scorer only from authorized upstream sources; add a formal citation or license only when the authors explicitly authorize it.

The timestamped pre-publication backups of the original small Git configuration files remain locally under ignored `archives/github_preparation_*` paths and were not uploaded.

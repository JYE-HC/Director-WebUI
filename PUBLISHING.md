# Publishing checklist for a first GitHub release

The clean `release/v0.1.0` branch is meant to be reviewed before it is pushed.
Director Web is licensed under GPL-3.0-only; bundled third-party components keep
their independently retained licenses.

1. Run `python tools/check_release.py` and the full backend/frontend test suite.
2. On real hardware, run one Standard generation and each RayLight topology you
   intend to advertise. Record only non-private results in `RELEASE.md`.
3. Create an empty GitHub repository without an auto-generated README or license.
4. Configure a repository-local GitHub noreply identity, then amend the release
   commit if you want that identity shown publicly:

   ```bash
   git config user.name "YOUR_GITHUB_NAME"
   git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
   git commit --amend --reset-author --no-edit
   ```

5. Add the remote and push only the clean release branch. Do not push the old
   private development repository, tags or backup refs:

   ```bash
   git remote add origin https://github.com/OWNER/REPOSITORY.git
   git push -u origin release/v0.1.0
   ```

6. Let GitHub Actions pass, including Gitleaks. A personal repository needs no
   extra configuration; an organization-owned repository may require a
   `GITLEAKS_LICENSE` secret. Review the rendered files, then merge or rename
   the branch to the desired default branch.
7. Create signed tag `v0.1.0` only after the GPU gates are complete.
   Attach a source archive and its SHA-256 digest to the GitHub Release.

Never use `git add -f` for local `data/`, `.data/`, logs, databases, media,
models, `.venv`, `node_modules`, maintainer notes or old Git history.

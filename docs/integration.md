# Git integration after review

The only temporary integration branch is `feature/evidence-binding-ablations` in the Proposed submodule. B1–B3 share this ablation repository and its ordinary history. Proposed `main` remains at the original commit until the changes are approved. No merge or push is performed by the implementation.

Inspect the exact local revisions:

```bash
git -C third_party/rcr rev-parse main
git -C third_party/rcr rev-parse feature/evidence-binding-ablations
git rev-parse HEAD
git submodule status
git -C third_party/rcr diff main...feature/evidence-binding-ablations --stat
```

After explicit approval, and after checking that Proposed `main` has not advanced, the minimal fast-forward workflow is:

```bash
git -C third_party/rcr fetch origin
git -C third_party/rcr switch main
git -C third_party/rcr merge --ff-only origin/main
git -C third_party/rcr merge --ff-only feature/evidence-binding-ablations
git -C third_party/rcr push origin main
```

If upstream `main` has diverged, stop the fast-forward workflow, merge the integration branch normally, resolve conflicts and rerun both CPU suites. Do not force-push or reset `main`. The feature branch can be published for a pull request instead if that is the chosen review workflow; that also requires approval.

After the approved merge, use the resulting **exact** Proposed commit for the ablation gitlink:

```bash
git -C third_party/rcr rev-parse HEAD
git add third_party/rcr
git diff --cached --submodule=short
git commit -m "Pin merged Proposed EvidenceBinding extension hooks"
git push origin main
```

A fast-forward preserves the integration commit SHA. If the ablation repository already pins it, `git add third_party/rcr` produces no change and the extra pin commit is unnecessary. A merge commit or conflict resolution produces a different SHA and needs the pin update. Publish Proposed first so new ablation clones can fetch the pinned commit, then publish the ablation repository. Both pushes above are documented steps for approval, not actions already taken.

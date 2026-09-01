# Frozen RoboMemArena compatibility inputs

Source: `OpenHelix-Team/RoboMemArena` at
`cc156e519990ae43cf3b64281a548724f428fbbd`.

Contents:

- `core/`: only files added or changed by RoboMemArena relative to this
  LIBERO checkout, plus its nested compatibility-package links;
- `bddl/`: official Task 1--26 BDDL files;
- `stage/`: official ordered-stage and shared physical pour-counter logic,
  with the broad evaluation-runner import replaced by a three-helper local
  adapter.

These files are evaluator-private. `scripts/robomemarena_bootstrap.py` overlays
them on a symlinked view of the current LIBERO package in the system temporary
directory. They must not be copied to an Agent workspace.

The frozen upstream commit did not contain a repository-root license file.
Resolve redistribution terms before publishing this directory outside the
research workspace.

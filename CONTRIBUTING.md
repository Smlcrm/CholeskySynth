# Contributing to CholeskySynth

Thanks for your interest in contributing. This repository currently centers on
the `cholesky_synth.ipynb` notebook prototype.

## Development workflow

1. Fork the repository (or create a branch if you have write access).
2. Branch off **`dev`** using the naming pattern `<author>/<slug>` or
   `chore/<slug>` (e.g. `deniz/add-periodic-kernel`).
3. Make focused changes; keep commits small and meaningful.
4. Open a pull request targeting **`dev`**. Release merges into **`prod`** are
   handled separately.

## Pull requests

- Give every PR a concise title and a short summary of *why* the change exists.
- Include a brief test plan (how a reviewer can verify the notebook still runs).
- Do not commit secrets, credentials, large binaries, or generated datasets —
  those belong in `.gitignore` / external storage.

## Code of conduct

By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please report vulnerabilities privately via the process in [SECURITY.md](SECURITY.md).
Do not open public issues for security reports.

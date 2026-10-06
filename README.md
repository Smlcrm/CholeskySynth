<!-- smlcrm:begin header -->
<!-- Logo: Smlcrm/design-system assets/logo/logo-digital-1.svg @ cf980f0. Light fill #2121a5 = token product.logo-ink; dark fill #ffffff = token brand-book.brand-white. -->
<p align="center">
  <a href="https://smlcrm.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/smlcrm-logo-dark.svg">
      <img alt="Simulacrum" src=".github/assets/smlcrm-logo-light.svg" width="300">
    </picture>
  </a>
</p>

<h1 align="center">CholeskySynth</h1>

<p align="center">A JAX notebook that generates synthetic multivariate time series by matrix-normal sampling over GP kernels.</p>

<p align="center"><a href="https://smlcrm.com">smlcrm.com</a></p>
<!-- smlcrm:end header -->

<!-- smlcrm:begin badges -->
<!-- Badge colours: 2121a5 = token brand-book.brand-dark-blue (license, language); 3483fa = token brand-book.brand-bright-blue (release). Public repository: license and release badges are dynamic. -->
<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/Smlcrm/CholeskySynth?color=2121a5"></a>
  <a href="https://github.com/Smlcrm/CholeskySynth/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Smlcrm/CholeskySynth?color=3483fa"></a>
  <!-- smlcrm:no-ci: the repository has no build or test workflow -->
  <a href="https://www.python.org/downloads/"><img alt="Language: Python 3.10+" src="https://img.shields.io/badge/python-%E2%89%A53.10-2121a5"></a>
</p>
<!-- smlcrm:end badges -->

<!-- smlcrm:begin overview -->
## Overview

CholeskySynth samples multivariate time series from a matrix-normal distribution. For each series it composes a temporal covariance and a cross-variate covariance from a bank of Gaussian-process kernels (constant, white noise, polynomial, RBF, rational quadratic, periodic), adds diagonal jitter, and draws the sample through their Cholesky factors. It generalises the univariate KernelSynth procedure, and the notebook can stream batches into a `tf.data.Dataset` to build large training sets.

**Who it is for.** Researchers and engineers who need large volumes of synthetic multivariate series with controllable temporal and cross-channel structure, for example to pre-train or test forecasting models.

**What it does not do.** It is a single Jupyter notebook, not an installable package or a command-line tool. It does not fit kernels to real data, train or evaluate models, or ship a pre-generated dataset. Cost grows cubically with series length and number of variates, so very long or very wide series need a GPU or smaller settings.
<!-- smlcrm:end overview -->

<!-- smlcrm:begin quickstart -->
## Quickstart

<!-- smlcrm:tested 2026-10-05 macOS 26.6 (Apple silicon, CPU), Python 3.11.14, jax 0.10.2, flax 0.12.8 -->
You need Python 3.10 or later and a clone of this repository. The sampler needs only JAX and Flax; the full notebook also uses the packages in `requirements.txt` and TensorFlow (see Installation below).

```bash
git clone https://github.com/Smlcrm/CholeskySynth.git && cd CholeskySynth
python -m venv .venv && . .venv/bin/activate
pip install jax flax
```

Save this as `quickstart.py` in the repository root and run `python quickstart.py`. It loads the notebook's kernel and sampling cells and draws four series of 256 steps and 3 variates on the CPU:

```python
import functools, json
from typing import Tuple
import jax
from jax import numpy as jnp
from flax import nnx

# Load the kernel bank and the sampler from the notebook, skipping its setup cells.
cells = json.load(open("cholesky_synth.ipynb"))["cells"]
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown" and "".join(cell["source"]).strip() in ("## Kernels", "## Sampling Methods"):
        exec("".join(cells[i + 1]["source"]))

key = jax.random.key(0)
ts, key = generate_matrix_normal_dataset(key, num_samples=4, num_rows=256, num_cols=3,
                                         num_time_kernels=4, num_variate_kernels=3, eps=1e-2)
print("shape:", ts.shape)
print("has NaNs:", bool(jnp.any(jnp.isnan(ts))))
```

Expected output:

```text
shape: (4, 256, 3)
has NaNs: False
```
<!-- smlcrm:end quickstart -->

## Features

- Generates long, multi-variate series with controllable temporal and cross-channel structure.
- Builds covariance factors by randomly composing Gram matrices derived from common kernels.
- Produces numerically stable samples by adding diagonal jitter before Cholesky factorization.
- Streams batches through a `tf.data.Dataset` writer for large-scale dataset creation.

## Repository Layout

- `cholesky_synth.ipynb` — end-to-end notebook containing all kernels, sampling routines, visualization helpers, and dataset exporters.
- `requirements.txt` — CPU-oriented dependencies for running the notebook.

## Prerequisites

- Python 3.10+ (tested with recent CPython/JAX releases).
- Conda, `venv`, or another environment manager is recommended to isolate dependencies.
- Latest NVIDIA drivers and CUDA 12.x if you intend to execute the GPU setup path.

## Installation

Create and activate an environment, then install the requirements. Two setup paths are provided in the notebook:

### CPU-only workflow

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install jax==0.6.2 tensorflow-cpu==2.17.0
```

### GPU workflow (CUDA 12)

```bash
pip install --upgrade pip
pip install "jax[cuda12]==0.6.2"
pip install "tensorflow[and-cuda]"
pip install -r requirements.txt
```

The notebook contains optional `%pip uninstall` steps that help clear conflicting GPU builds; run them only if you need to switch backends on a shared machine.

## Getting Started

1. Launch JupyterLab or VS Code and open `cholesky_synth.ipynb`.
2. Run the environment setup cell that matches your hardware (CPU or GPU).
3. Execute the remaining cells to import libraries, define kernels, generate samples, and optionally persist datasets.

A minimal Python snippet from the notebook illustrates how to draw a batch of matrix-normal samples:

```python
import jax
from jax import random

key = random.key(0)
batch_size = 10        # number of series
length = 2500          # time steps per series
num_variates = 10      # channels per series
num_time_kernels = 4   # temporal kernel components
num_variate_kernels = 3
eps = 1e-2             # diagonal jitter for stability

ts_batch, key = generate_matrix_normal_dataset(
    key,
    num_samples=batch_size,
    num_rows=length,
    num_cols=num_variates,
    num_time_kernels=num_time_kernels,
    num_variate_kernels=num_variate_kernels,
    eps=eps,
)
```

Use `plot_time_series_grid(ts_batch)` to visualize generated trajectories.

## Algorithm Overview

CholeskySynth minimizes the simulator objective described in Equation (1) of the accompanying manuscript by sampling multivariate time series from a matrix-normal distribution:

1. **Kernel bank** — Temporal (`U`) and cross-variate (`V`) covariances are constructed from Gram matrices computed with constant, white-noise, polynomial, RBF, rational-quadratic, and periodic kernels at multiple hyper-parameter settings.
2. **Random convolution** — For each draw, the notebook samples a subset of Gram matrices for both `U` and `V` and recursively combines them with random additive or multiplicative operators to emulate kernel composition.
3. **Numerical stabilization** — A configurable jitter `eps` is added to the diagonals of `U` and `V` to guarantee positive definiteness prior to Cholesky factorization.
4. **Matrix-normal sampling** — Standard normal noise is transformed with the Cholesky factors `A` and `B`, yielding samples with covariance `U ⊗ V`. This provides an efficient stand-in for a general matrix-normal sampler, which is not available off-the-shelf in JAX.

The implementation scales cubically with both the number of time steps and variates due to the Cholesky decompositions, with quadratic memory usage. Adjust `length`, `num_variates`, and kernel counts carefully when targeting very large batches.

## Data Generation Pipeline

- `create_data_batch` wraps `generate_matrix_normal_dataset` and guards against NaNs by regenerating batches if necessary.
- `data_generator` streams batches with randomized kernel counts into a `tf.data.Dataset`, enabling asynchronous prefetching and on-disk persistence.
- Batches are saved with timestamped directories under `../data/tempo_v1_largest_*` by invoking `dataset.save(...)`. Ensure the parent directory exists when running outside the notebook root.

To export a million samples one series at a time, adjust `batch_size`, `length`, and the kernel ranges near the bottom of the notebook before executing the save cell.

## Customization Tips

- **Kernel diversity**: Extend `compute_all_gram_matrices` with additional kernels or alternate hyper-parameter grids to broaden the covariance family.
- **Stability vs. fidelity**: Increase `eps` for more aggressive jitter if you encounter decomposition failures; decrease it to retain sharper correlations once the setup is stable.
- **Performance**: Leverage GPUs (or TPUs via JAX) when generating very large datasets. Consider reducing `num_time_kernels`/`num_variate_kernels` to shorten kernel composition chains.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes. Please also read our [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## Acknowledgements

CholeskySynth is a multivariate extension of KernelSynth [7], relying on properties of the matrix-normal distribution [9]. The implementation here packages the sampling routine, kernel compositions, and dataset writer in a single notebook for reproducibility and reuse.

---

For questions or contributions, please open an issue or submit a pull request.

<!-- smlcrm:begin links -->
## Links

- Documentation: this README and the notebook [`cholesky_synth.ipynb`](cholesky_synth.ipynb)
- Issues: [Smlcrm/CholeskySynth/issues](https://github.com/Smlcrm/CholeskySynth/issues)
- Releases: [Smlcrm/CholeskySynth/releases](https://github.com/Smlcrm/CholeskySynth/releases)
- Website: [smlcrm.com](https://smlcrm.com)
- Related repositories: none
<!-- smlcrm:end links -->

<!-- smlcrm:begin citation -->
## Citation

If you use CholeskySynth in your work, cite it as below. GitHub's "Cite this repository" button reads the same data from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{smlcrm_cholesky_synth,
  title   = {CholeskySynth},
  author  = {{Simulacrum, Inc.}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Smlcrm/CholeskySynth}
}
```
<!-- smlcrm:end citation -->

<!-- smlcrm:begin license -->
## License and contact

Released under the MIT license. See [LICENSE](LICENSE).

Contact: [support@smlcrm.com](mailto:support@smlcrm.com) · [smlcrm.com](https://smlcrm.com) · [github.com/Smlcrm](https://github.com/Smlcrm)
<!-- smlcrm:end license -->

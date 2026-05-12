# kdtree-rs

A minimal, high-performance KDTree for free-threaded CPython.

## Install

```bash
pip install kdtree-rs
# or
uv add kdtree-rs
```

Requires Python 3.13+ with free-threaded build (`cp313t` / `cp314t`).

## Usage

The PyPI distribution `kdtree-rs` is imported as `kdtree`:

```python
import numpy as np

from kdtree import KDTree

data = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
tree = KDTree(data, leafsize=32)

distances, indices = tree.query(np.array([[0.2, 0.0], [2.8, 0.0]]), k=2)
```

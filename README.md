# kdtree

A minimal, high-performance KDTree for free-threaded CPython.

```python
import numpy as np

from kdtree import KDTree

data = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
tree = KDTree(data, leafsize=32)

distances, indices = tree.query(np.array([[0.2, 0.0], [2.8, 0.0]]), k=2)
radius_hits = tree.query_radius(np.array([0.2, 0.0]), 1.0, sort=True)
pairs = tree.query_pairs(1.1)
```

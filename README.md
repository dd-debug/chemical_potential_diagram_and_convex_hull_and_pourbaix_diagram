## Overview

Here we offer packages to visualize and analyze high dimensional chemical potential diagram, convex hull phase diagrams, and Pourbaix diagrams.

## Prerequisites

### Pymatgen

These packages have a dependency on `pymatgen` package of the Materials Project database using Python 3. You can install `pymatgen` by either

1. install the required packages in requirements.txt

   ```bash
   pip install -r requirements.txt
   ```

2. Go [here](https://pymatgen.org/installation.html) and follow the instructions to install your `pymatgen`.

### Pymatgen API Key

To use pymatgen, you need to generate an API key. This algorithm is using the legacy Materials Project API by default, but you can switch to new Materials Project API if needed.

- Go [here](https://legacy.materialsproject.org/open) to get a legacy Materials Project API
- Go [here](https://next-gen.materialsproject.org/api) to get a new Materials Project API

After you get a API Key, copy it and go to `phase_diagram_packages/convexhullpdplotter.py` python file,  paste its string to `getOrigStableEntriesList` function:

```python
with MPRester("Your API key") as MPR:
```

## Tutorials and examples

We offer examples to generate 3d convex hull phase diagram, and plot tangent planes or reaction compound convex hulls; and also binary/ternary chemical potential diagrams, mixed chemical potential and composition phase diagrams; high dimensional Pourbaix diagram with arbitrary axis, such as pH, E, chemical potential, particle radius, etc. 

Besides the example we offered here, these packages can actually do more phase diagrams, feel free to contact jiadongc@umich.edu, if you have any new plan on making new phase diagrams using these codes.

   

## Citation

These packages are created by Jiadong Chen, Wenhao Sun in University of Michigan. If you use packages, we kindly ask you to cite the following publication:

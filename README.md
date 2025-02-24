# DAMCC
### (A Deep Autoregressive Model for Dynamic Combinatorial Complexes)

Find the original paper [in this link](
https://zenodo.org/records/14907028?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE3ZTA0OGQ0LTZkZTEtNDAwMy1iZmY2LTA5MTBhYTM2MmNhYyIsImRhdGEiOnt9LCJyYW5kb20iOiI4ZDFlMGRhMTU2NjMxOThmYzZkNTJkYWFlOGU1N2MwNCJ9.pz70sQyjHbS7mQseL7uqVm1x8ZNiYIZuYuyqS9wEUFmhCVEz0V7SfAAawDY3nyil1MJwQLC7aXVGYOjc0rNnyg).

## Dependencies
Please install the packages in this order as instructed.

This project was developed in  Python 3.11.3, so we recommend this version.

Create a conda environment (or an environment of your choice):
   ```bash
   conda create -n damcc python=3.11.3
   conda activate damcc
   ```
The first dependency is TopoModelX.
### Installing TopoModelX

`TopoModelX` is available on PyPI and can be installed using `pip`.
Run the following command:

```bash
pip install topomodelx
```

Then install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.
```bash
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```
where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).
Then run
```bash
pip install -r requirements.txt
```
For a quick tutorial see notebooks/damcc_tutorial.ipynb
# tools

* `make_figures.py` rebuilds `docs/images/*.png` from `figure_data.json` (needs matplotlib only).
* `collect_figure_data.py` regenerates `figure_data.json`. It needs the research artifacts:
  * a local `facebook/opt-125m` folder;
  * the true input covariances, measured on held-out WikiText-2 windows (evaluation only);
  * the `results.json` files written by `tests/real_models.py`.

# What Drives Short-Term Stock Volatility?

Final project for STA 5650 (Multivariate Data Analysis). We use the Kaggle
[Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
data: 112 anonymized stocks, 3,830 ten-minute windows, and about 429,000 rows.

## Research question

Over 10-minute windows, is realized volatility across 112 stocks driven by a
small number of common factors? And after accounting for each stock's own recent
volatility and those common factors, do order book features (bid-ask spread,
market depth, order imbalance, trading activity) still add predictive power?

## Hypotheses

- **H1, common factor:** One or two principal components explain a large share
  of the variation in log volatility, and stocks fall into a few natural groups
  based on their factor loadings.
- **H2, group differences:** These groups differ in their order book profiles.
- **H3, incremental value:** Starting from a stock's own recent volatility,
  adding the common factor and then order book features each reduces forecast
  error.
- **H4, honest evaluation:** Random cross-validation overstates forecast
  accuracy compared with a time-ordered split.

## Team roles

| Role | Focus | Python notebooks | R scripts |
| --- | --- | --- | --- |
| A | Data & EDA | `01_features` | `03_eda` |
| B | Multivariate structure (H1, H2) | | `05_pca_fa_cluster`, `06_manova` |
| C | Modeling & validation (H3, H4) | `02_time_order`, `07_lightgbm_benchmark` | `04_acf`, `07_regression` |

## Which language for what

| Step | Language | File |
| --- | --- | --- |
| 1. Build features | Python | `notebooks/01_features.ipynb` |
| 2. Recover time order | Python | `notebooks/02_time_order.ipynb` |
| 3. EDA | R | `R/03_eda.R` |
| 4. ACF | R | `R/04_acf.R` |
| 5. PCA, factor analysis, clustering | R | `R/05_pca_fa_cluster.R` |
| 6. MANOVA and profile analysis | R | `R/06_manova.R` |
| 7. Regression models | R | `R/07_regression.R` |
| 7. LightGBM benchmark | Python | `notebooks/07_lightgbm_benchmark.ipynb` |

### The handoff rule

Python writes `data/processed/features.parquet`, and R reads it with arrow.
**Nothing else crosses between the two.**

So everything R needs from Python must be a column in that file:

- `01_features` attaches the target from `train.csv`.
- `02_time_order` adds the `time_rank` column. Rerun it whenever you rerun
  `01_features`.

R scripts can share their own files with each other, such as
`stock_clusters.parquet`. Python never reads those files.

## Python setup

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install
```

`nbstripout --install` removes notebook outputs when you commit, which keeps the
repo small and prevents merge conflicts. Run it once in each clone.

On a Mac, LightGBM may also need `brew install libomp`.

Check that everything works:

```bash
pytest
```

## R setup

1. Open `optiver-volatility-sta5650-project.Rproj` in RStudio. This sets the
   working directory to the project root and starts renv.
2. Install the exact package versions we all share:

   ```r
   renv::restore()
   ```

   If that fails, run `source("R/packages.R")` instead. It installs any missing
   packages, but their versions may differ from `renv.lock`.

On a Mac, `heplots` (used for Box's M) depends on `rgl`, which needs
[XQuartz](https://www.xquartz.org). If `library(heplots)` fails with an error
about `/opt/X11`, run `brew install --cask xquartz`, then log out and back in.

If you add a new package, add it to `R/packages.R`, install it, run
`renv::snapshot()`, and commit the updated `renv.lock`.

## Get the data

**Never commit the data.** It is too large for GitHub, and Kaggle's rules
restrict sharing it. The `.gitignore` blocks it, but please don't force-add it.

1. Accept the rules on the
   [competition page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/rules).
2. Create an API token on your Kaggle account page and put `kaggle.json` in
   `~/.kaggle/`. Keep this file private.
3. Download and unzip the data:

   ```bash
   kaggle competitions download -c optiver-realized-volatility-prediction -p data/raw
   unzip data/raw/optiver-realized-volatility-prediction.zip -d data/raw
   ```

The download is about 1.7 GB, and it is larger once unzipped.

## Running heavy jobs on Kaggle

Feature extraction reads every stock's raw files, so it can be slow on a laptop.
You can run it on Kaggle Notebooks instead, where the data is already available.
The team then only needs to share the small `features.parquet`, which goes in
`data/processed/`. Share it through a private channel, not through this repo.
Run `02_time_order` before sharing, so that the file includes `time_rank`.

## Team workflow

- **One owner per notebook or R script.** Only the owner edits it, which avoids
  merge conflicts. Ask the owner if you need a change.
- **Run `git pull` before you start.**
- **Work on your own branch**, then open a pull request into `main`:

  ```bash
  git checkout main
  git pull
  git checkout -b your-name/short-description
  # Do your work, then add only the files you changed.
  git add path/to/changed_file
  git commit -m "Describe your change"
  git push -u origin your-name/short-description
  ```

- Put shared Python code in `src/` and add tests in `tests/`. To import it from
  a notebook:

  ```python
  import sys
  sys.path.append("..")

  from src.features import build_stock_features
  from src.metrics import rmspe
  ```

- Put shared R code in `R/utils.R`. R scripts load it with
  `source("R/utils.R")`, then read the data with `load_features()`.

Notebook outputs are not committed, so save anything you need to keep in
`figures/` or `report/`. CSV files are also ignored, so save result tables as
Markdown.

## Repo layout

```
data/raw/         Kaggle files (not committed)
data/processed/   Files we generate, such as features.parquet (not committed)
notebooks/        Python notebooks: features, time order, LightGBM benchmark
src/              Shared Python code
tests/            pytest tests for src/
R/                R analysis scripts, shared helpers, and the package list
renv/, renv.lock  Pinned R package versions
figures/          Saved plots
report/           Final report and result tables
```

Run the steps in numbered order. Run R scripts from the project root.

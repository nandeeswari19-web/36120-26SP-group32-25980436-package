# nba-draft-features

Reusable data-cleaning and feature-engineering utilities for predicting whether a US college
basketball player will achieve a **sustained NBA career** (≥ 3 NBA seasons), built as part of
UTS 36120 Assignment Task 1 (2026 Spring), student 25980436, group 32.

This package exists so that the same tested logic used across my experiment notebooks
(`36120-26SP-group32-25980436-AT1-experiment-*.ipynb` in the group experimentation repository)
is written once, unit-tested, and imported — rather than copy-pasted between notebooks.

## Installation

From TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ nba-draft-features
```

For local development:

```bash
git clone https://github.com/<your-github-username>/36120-26SP-group32-25980436-package.git
cd 36120-26SP-group32-25980436-package
pip install -e ".[dev]"
pytest
```

## Usage

```python
import pandas as pd
from nba_draft_features import (
    clean_height,
    clean_class_year,
    add_age_at_season,
    add_per40_rates,
    add_shot_profile_ratios,
    add_interaction_features,
)

df = pd.read_csv("train.csv")
df = clean_height(df)              # adds `ht_in` from "6-7" style strings
df = clean_class_year(df)          # adds `class_yr_ord` (Fr=1 ... Sr=4)
df = add_age_at_season(df)         # adds `age_season`, drops raw `dob`
df = add_per40_rates(df)           # adds `pts_per40`, `treb_per40`, `ast_per40`
df = add_shot_profile_ratios(df)   # adds `rim_rate`, `dunk_rate`
df = add_interaction_features(df)  # adds `usage_efficiency`, `defensive_versatility`
```

## Modules

| Module | Contents |
|---|---|
| `nba_draft_features.cleaning` | `ht_to_inches`, `clean_height`, `clean_class_year` |
| `nba_draft_features.features` | `add_age_at_season`, `add_per40_rates`, `add_shot_profile_ratios`, `add_interaction_features` |

## Testing

```bash
pytest tests/ -v
```

## Publishing (TestPyPI)

```bash
python -m build
python -m twine upload --repository testpypi dist/*
```

## License

MIT

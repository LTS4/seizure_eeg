Seizure learning
==============================

Analysis and development of methods for seizure detection and prediction


## Data processing

### TUH data download

Instruction for downloading the TUH seizure corpus can be found on the [TUH EEG Corpus website][tuh_web].
After registration, you will get a password for the `nedc` username.

We provide an automated script for downloading the data in `rsync_nedc.sh`.
It keeps querying the server until the download of the desired corpus is complete.
It expects two arguments: `SOURCE` and `TARGET`, e.g.
```sh
# rsync_nedc.sh SOURCE TARGET
bash rsync_nedc.sh tuh_eeg_seizure/v1.5.2 data/raw/TUSZ
```

It then asks for the password.
To avoid re-prompting the password continuously, we also provide the `rsync_answer.exp` script.
In addition to `SOURCE` and `TARGET` it expects the `NEDC_PASSWORD`:
```sh
# expect rsync_answer.exp SOURCE TARGET PASSWORD
expect rsync_answer.exp tuh_eeg_seizure/v1.5.2 data/raw/TUSZ password1234
```

If you get a `"Permission denied, please try again."` message it is probably because your password is wrong.

[tuh_web]: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

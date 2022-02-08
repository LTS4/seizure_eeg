================================
EEG pipeline for reproducible ML
================================


To open the road for reproducibility, we implement a parametrized preprocessing library providing the functionality required to extract clips in the format required by many ML algorithms.
Currently, it is implemented for the TUH corpus, and other datasets will be integrated soon.
To simplify adoption, we focused on using well know and performant Python libraries, such as
``pandas``_, ``numpy`` and ``pytorch``.

.. _``pandas``: https://pandas.pydata.org/

Data processing
===============

TUH data download
=================

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


Code structure
==============
    .
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── config.yaml        <- Example configuration file with paths and options for data loading and
    │                         preprocesing
    ├── pyproject.toml
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── seiz_eeg
    │   ├── __init__.py
    │   ├── config.py
    │   ├── dataset.py
    │   ├── schemas.py
    │   └── tusz
    │       ├── __init__.py
    │       ├── annotations
    │       │   ├── __init__.py
    │       │   ├── io.py
    │       │   └── process.py
    │       ├── constants.py
    │       ├── download.py
    │       ├── io.py
    │       ├── main.py
    │       ├── process.py
    │       ├── signals
    │       │   ├── __init__.py
    │       │   ├── io.py
    │       │   └── process.py
    │       └── utils.py
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported

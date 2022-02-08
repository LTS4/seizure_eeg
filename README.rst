================================================================================
EEG pipeline for reproducible ML
================================================================================


To open the road for reproducibility, we implement a parametrized preprocessing
library providing the functionality required to extract clips in the format
required by many ML algorithms.  Currently, it is implemented for the TUH
corpus, and other datasets will be integrated soon.  To simplify adoption, we
focused on using well know and performant Python libraries, such as pandas_,
numpy_, scipy_, and pytorch_.

.. _pandas: https://pandas.pydata.org/
.. _numpy: https://numpy.org/
.. _pytorch: https://pytorch.org/
.. _scipy: https://scipy.org/

How to use
================================================================================

This package provides the following functionalities:

1. Data fetching
2. Pre-processing of EEG measurements
3. Creation of clips dataframe, with relevant start and end times
4. Pytorch dataset implementation, which handles:

   - Data loading
   - Data transforms

The first two steps are handled by ``seiz_eeg.preprocess``, whose parameters can
be set in the ``data_config.yaml`` file, or passed as cli arguments.
A ``dataset`` must be specified with the corresponding option, either in the
``.yaml`` file, or as follows::

    python -m seiz_eeg.preprocess dataset=tusz

The creation of clips and the Pytorch Dataset are implemented in
``seiz_eeg.dataset``.

Download and pre-processing
--------------------------------------------------------------------------------

Data are downloaded to a subfolder of ``raw_path``, declared in the
source-specific configuration.
Then, with functions which are tailored to different datasets, we pretreat the
the data to give them a source agnostic structure.

Segments dataframe
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

EEG measurements come with little structure. To perform any data-driven task, we
shall identify relevant information and organize them.
This is generally provided in annotations files, which are separate for each
EEG scan. By preliminary reading all of such files, we can create a tabular
*annotation dataframe*, where entries are indexed by patient, session,
segment, and channel. The following image shows a sample of such
a table for the training split of the TUH Seizure corpus. Thanks to this added
structure, it is easy to define *clips of interest* and quickly retrieve the
relevant signals file, which can be read and processed.

.. image:: docs/figures/segments-df.png
   :alt: Sample of an annotation dataframe
   :width: 100 %
   :align: center


EEG signals
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. image:: docs/figures/processing.png
   :alt: Schema of preprocessing pipeline
   :width: 100 %
   :align: center

Usual pre-processing of EEG signals. We read raw signals from a ``.edf`` file
and resample them to the desired rate. Then we extract the clip of interest,
e.g. the first seconds of a seizure, and we optionally split it in windows.
Those can then be further transformed or fed to a model. Since many clips can be
extracted out of the same file, it is convenient to save them and avoid
repeating expensive operations.

Datasets
================================================================================

TUH Seizure corpus
--------------------------------------------------------------------------------

This corpus consists in many hours of labelled EEG sessions.
The ``seiz_eeg.tusz`` module provides code specific to this dataset annotations
and EEG measurements.

To download the data, you need to register (free account).
You will get a password for the ``nedc`` username.
The password shall be included in the ``data_config.yaml`` file, or passed to
the command line as follows:

.. code-block:: sh

    python -m seiz_eeg.preprocess dataset=tusz tusz.password=$PASSWORD

If you get a ``"Permission denied, please try again."`` message it is probably
because your password is wrong.

More information about the TUH seizure corpus can be found on the `TUH EEG
Corpus website`_.

.. _`TUH EEG Corpus website`:
    https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml


Code structure
================================================================================

.. code-block::

    .
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this
    │                         project.
    ├── config.yaml        <- Example configuration file with paths and options
    │                         for data loading and preprocesing
    ├── pyproject.toml
    │
    ├── docs               <- Folder containing Sphinx directives and figures
    │
    ├── seiz_eeg
    │   ├── __init__.py
    │   ├── config.py
    │   ├── dataset.py
    │   ├── schemas.py
    │   └── tusz
    │       ├── __init__.py
    │       ├── annotations
    │       │   ├── __init__.py
    │       │   ├── io.py
    │       │   └── process.py
    │       ├── constants.py
    │       ├── download.py
    │       ├── io.py
    │       ├── main.py
    │       ├── process.py
    │       ├── signals
    │       │   ├── __init__.py
    │       │   ├── io.py
    │       │   └── process.py
    │       └── utils.py
    │
    └── setup.py           <- Options for package building

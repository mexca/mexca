Installation
============

**Mexca** supports Python >=3.7 and Python <= 3.9. We recommend installing mexca via the terminal/command prompt.

Installation Steps on Windows
-----------------------------

Open the terminal/command prompt (by right-clicking the Windows icon in the bottom-left corner of your screen, or with the keyboard shortcut `Windows Key` + `X`). We recommend to install mexca in a new virtual environment, e.g., using `venv`, so type the following in the terminal:

.. code-block:: console

  python3 -m venv mexca-venv
  env/bin/activate

Alternatively, if you use conda:

.. code-block:: console

  conda create -n mexca-venv
  conda activate mexca-venv

Once you have activated your virtual environment you can then install **mexca** from PyPi:

.. code-block:: console

  python3 -m pip install mexca

Alternatively, you can install **mexca** from GitHub via:

.. code-block:: console

  git clone https://github.com/mexca/mexca.git
  cd mexca
  python3 -m pip install .

Or via:

.. code-block:: console

  python3 -m pip install git+https://github.com/mexca/mexca.git

Installation Steps on Unix/macOS
--------------------------------

Open the terminal (click the Launchpad icon in the Dock, type “Terminal” in the search field; otherwise, you can use the keyboard shortcut `Command` + `Space`, and type in “Terminal”).

We recommend to install mexca in a new virtual environment, e.g., using `venv`, so type the following within the terminal:

.. code-block:: console
  python3 -m venv mexca-venv
  source env/bin/activate

Alternatively, if you use conda:

.. code-block:: console
  conda create -n mexca-venv
  conda activate mexca-venv

Once you have activated your virtual environment you can then install **mexca** from PyPi:

.. code-block:: console

  python3 -m pip install mexca

Alternatively, you can install **mexca** from GitHub via:

.. code-block:: console

  git clone https://github.com/mexca/mexca.git
  cd mexca
  python3 -m pip install .

Or via:

.. code-block:: console

  python3 -m pip install git+https://github.com/mexca/mexca.git

Issues installing mexca for M1 Macbook users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many deep learning libraries that we import in mexca do not fully support the Apple M1 yet, which can lead to several issues when installing mexca. We provide few workarounds for the most common issues. They have been tested on Python 3.9.0 in a conda environment (last update 3/10/2022).

Error n. 1: 

- OSError cannot load libsndfile.dylib (Github issue [#311](https://github.com/bastibe/python-soundfile/pull/311)):

.. code-block:: console
  
  OSError: cannot load library '...venv/lib/python3.9/site-packages/_soundfile_data/libsndfile.dylib': dlopen(...venv/lib/python3.9/site-packages/_soundfile_data/libsndfile.dylib, 2): image not found

To fix this:

1. Make sure that you have installed libsndfile via brew, if not [install it](https://formulae.brew.sh/formula/libsndfile). 
2. Copy the libsndfile installed from Homebrew (/opt/homebrew/lib/_soundfile_data/libsndfile.dylib) into the expected folder ‘python3.9/site-packages/_soundfile_data/‘ 
3. Restart the kernel.

Error n. 2: 

- OSError cannot load libllvmlite.dylib (Github issue [#650](https://github.com/numba/llvmlite/issues/650)):

.. code-block:: console

  OSError: Could not load shared object file: libllvmlite.dylib

To fix this:

1. Type in the terminal:

.. code-block:: console

  conda install -c numba numba
  conda install -c numba llvmlite

2. Restart the kernel.

*TIP:* Make sure to run those fixes in the terminal, or in the jupyter notebook in a cell preceded by the symbol '!'. Make sure that the activated environment you're running the fixes is the one where you are attempting to install mexca (i.e., if you followed the installation steps above, it will be 'mexca-venv').

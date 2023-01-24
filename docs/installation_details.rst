Installation Details
====================

Installing Python
-----------------

The easiest way to install Python on your system is via `Anaconda <https://www.anaconda.com/products/distribution>`_.
To check if the system has a Python distribution installed, open a terminal window and run:

.. code-block:: console

    python --version

or for Python 3 specifically:

.. code-block:: console

    python3 --version


Opening a Terminal Window
-------------------------

On Windows, press the Windows key and search for "PowerShell". On macOS, press Cmd + Space and search for "Terminal".


Creating a Virtual environment
------------------------------

If Python was installed via Anaconda, a virtual environment can be created via the Terminal with `conda`:

.. code-block:: console

    conda create -n mexca-venv 'Python<3.10'

Then, activate the environment with:

.. code-block:: console

    conda activate mexca-venv

Alternatively, you can create a virtual environment with `venv`:

.. code-block:: console

    python -m venv mexca-venv

On Windows, the `venv` environment can be activated with:

.. code-block:: console

    mexca-venv\Scripts\activate

On macOS and Linux, run:

.. code-block:: console

    source mexca-venv/bin/activate


Installation
------------

Before installing mexca, make sure, all other :ref:`Requirements` are installed.

After creating and activating the virtual environment, run ``pip install mexca`` in the Terminal to install the mexca base package from PyPI. 
This will only install the dependencies of the base package. The pipeline components can still be run as Docker containers.

.. note::
    
    This setup requires that Docker is installed on your system.

To run the components without containers, their additional dependencies must be installed via:

.. code-block:: console

    pip install mexca[vid,spe,voi,tra,sen]

The abbreviations indicate:

* `vid`: FaceExtractor
* `spe`: SpeakerIdentifier
* `voi`: VoiceExtractor
* `tra`: AudioTranscriber
* `sen`: SentimentExtractor

.. note::

    It is also possible to run some pipeline components with containers and others without.
    For example, the requirements for only the FaceExtractor can be installed via ``pip install mexca[vid]``


Installing the Development Version
----------------------------------

For the latest features and bug fixes, the development version of mexca can be installed from GitHub (requires `Git <https://git-scm.com/>`_) via:

.. code-block:: console

    pip install git+https://github.com/mexca/mexca.git

This command will install the latest developments of mexca on the main branch.


Running Example Notebooks
-------------------------

The mexca GitHub repository contains `Jupyter <https://jupyter.org/>`_ notebooks with examples. This requires that Jupyter is installed, which can be done via:

.. code-block:: console

    pip install notebook

To run notebooks in your virtual environment, ipykernel also needs to be installed:

.. code-block:: console

    pip install ipykernel

Clone the repository with `Git <https://git-scm.com/>`_:

.. code-block:: console

    git clone https://github.com/mexca/mexca.git
    cd mexca # go to package directory

To start Jupyter, run:

.. code-block:: console

    jupyter notebook

Select an example notebook in the ``examples/`` folder.

Installation on M1 Devices
--------------------------

PyTorch offers beta support for M1 devices since version 1.13. mexca, however, specifies PyTorch version 1.12 as a requirement for its components.
To run mexca on M1 devices, upgrading PyTorch to 1.13 is a potential solution, but currently not tested regularly. This can be done via:

.. code-block:: console

    pip install torch==1.13

.. note::
    Currently, components **cannot** be run as containers on M1 devices.

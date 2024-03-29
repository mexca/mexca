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


Creating a Virtual Environment
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

All five components can be installed via:

.. code-block:: console

    pip install mexca[all]

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

Installation with CUDA
----------------------

To run mexca on a GPU, PyTorch needs to be installed with CUDA. Note that this does not work with containerized components.
If not already installed, download and install the appropriate `CUDA toolkit <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_ for your operating system. Then, install PyTorch with CUDA support:

.. code-block:: console

    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

It is important that the installed version of the CUDA toolkit matches supported version by PyTorch.
In this example, both versions point to CUDA 11.8. If a different CUDA version is installed on your system, you need to install the matches PyTorch version. See `this <https://pytorch.org/get-started/previous-versions/>`_ link to find the correct version.

Troubleshooting
---------------

This section mentions some reoccuring issues and how to solve them.

Install pypiwin32 package
^^^^^^^^^^^^^^^^^^^^^^^^^

When running mexca on Windows and depending on the Python distribution, this error can occur when running containerized components for the first time:

.. code-block:: console

    docker.errors.DockerException: Install pypiwin32 package to enable npipe:// support

A solution to this problem is running the pypiwin32 postinstall script manually. When mexca was installed in a virtual environment, this can be done via:

.. code-block:: console

    python mexca-venv\Scripts\pywin32_postinstall.py -install

Otherwise, search for the location of the script and run it from there.

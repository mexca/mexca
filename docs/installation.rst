Installation
============

**Mexca** can be installed via github (see below). Installing **mexca** will install automatically all the Python-based components of the system, i.e., the core, video, audio and text pipelines along with their dependencies. This requires Python >= 3.7 and Python <= 3.9.12.

It is usually a good idea to install **mexca** into a new virtual environment, for example using venv:

.. code-block:: console

  python3 -m venv mexca-venv
  mexca_venv/bin/activate

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

Installation
============

Python
------

**Mexca** is currently under development, and can be installed via github (see below). Installing **Mexca** will install automatically all the Python-based components of the system, i.e., the core, video, audio and text pipelines along with their dependencies. This requires at least Python >= 3.7 and Python <= 3.9.12

It's normally a good idea to make a virtual environment (virtualenv) within which to install it. If you don't have one:

.. code-block:: console

  ~$ python3 -m venv mexca_venv
  ~$ mexca_venv/bin/activate


Alternatively, if you use conda:
.. code-block:: console

  ~$ conda create -n mexca_venv
  ~$ conda activate mexca_venv

Once you've activate your virtual environment you can then install mexca from the Github repository in this way:

.. code-block:: console

  ~$ python3 -m pip install git+https://github.com/mexca/mexca.git

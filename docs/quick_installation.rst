Quick Installation
==================

In this section, we explain briefly how to install mexca on your system. Detailed instructions can be found in the :ref:`Installation Details` section.
mexca can be installed on Windows, macOS and Linux. We recommend Windows 10, macOS 12.6.x, or Ubuntu.

.. note::

    The package contains five components that must be explicitly installed [#]_. By default, only the base package is installed
    (which requires only a few dependencies). The components can still be used through Docker containers which must be downloaded
    from Docker Hub. We recommend this setup for users with little experience with installing Python packages or who simply want to
    quickly try out the package. Using the containers also adds stability to your program.

Requirements
------------

mexca requires Python version >= 3.8 and <= 3.10. It further depends on `FFmpeg <https://ffmpeg.org/>`_ (for video and audio processing),
which is usually automatically installed through the MoviePy package (i.e., its imageio dependency). In case the automatic install fails,
it must be installed manually.

To download and run the components as Docker containers, Docker must be installed on your system. Instructions on how to install
Docker Desktop can be found `here <https://www.docker.com/get-started/>`_.

All components but the VoiceExtractor depend on PyTorch (version 1.12). Usually, it should be automatically installed when specifying any
of these components. In case the installation fails, see the installation instructions on the PyTorch `web page <https://pytorch.org/get-started/locally/>`_.

For the SpeakerIdentifier component, the library `libsndfile <https://libsndfile.github.io/libsndfile/>`_ must also be installed on Linux systems.

The SentimentExtractor component depends on the `sentencepiece <https://github.com/google/sentencepiece>`_ library,
which is automatically installed if `Git <https://git-scm.com/>`_ is installed on the system.

Installation
------------

We recommend installing mexca in a new virtual environment to avoid dependency conflicts. The base package can be installed from PyPI via `pip`:

.. code-block:: console

    pip install mexca

The dependencies for the additional components can be installed via:

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

To run the demo and example notebooks, install the Jupyter requirements via:

.. code-block:: console

    pip install mexca[demo]

Requirements related to package development (e.g., running tests) or publication can be installed via:

.. code-block:: console

    pip install mexca[dev,publishing]

.. [#] We explain the rationale for this setup in the :ref:`Docker` section.

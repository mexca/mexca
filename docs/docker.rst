Docker
======

Running a Container
-------------------

The **mexca** pipeline is available as a `Docker image <https://hub.docker.com/repository/docker/mluken/mexca>`_. To run the pipeline via Docker, `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ needs to be installed on the computer and active.
The pipeline can be used via the command line:

.. code-block:: console

  docker run -t -v absolute/path/to/video_file:/mexca mexca:stable -f video_file -o output.json

This command pulls the image with the latest release of **mexca**, executes a script to run the pipeline on a video file, and save the output on the computer.

*Explanation of the command*: 

- The `-t` flag indicates that a specific tag of the mexca Docker image should be run. Currently, two tags are avaiable, 'stable' (the latest release version) and 'devel' (the development version). It is also possible to run the image with the latest tag using `mexca:latest`. 
- With the `-v` flag, a folder from the host system is mounted onto the container. Here, the absolute path to the folder that should be mounted must be supplied as well as the destination folder in the container (here `/mexca`). 
  This allows the container to access the video file to which the pipeline will be applied to and write the output on the host system. 
- This is followed by the name of the image and the tag (`mexca:stable`).
- By default, the container runs the `bin/pipeline.py` script, so the command line arguments for the script must be given after all `docker run` options. For details on the arguments, see the documentation of the `pipeline.py` script.

Building an Image
-----------------

The **mexca** image can be build using the command line:

.. code-block:: console

  git clone https://github.com/mexca/mexca.git
  cd mexca
  docker build --build-arg version=stable . -t mexca:stable

First, the GitHub repository is cloned to provide the files that are necessary for building the image. Different versions can be build using the `version` argument (here the 'stable' version). To build the development version, run:

.. code-block:: console

  docker build --build-arg version=devel . -t mexca:devel

For details on the build, see the `Dockerfile` in the **mexca** repository.

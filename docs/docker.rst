Docker
======

Rationale for Containerized Components
--------------------------------------

The pipeline implemented in mexca is complex and requires many large dependencies. It also uses pretrained models to perform many tasks. The weights of
these models must be downloaded before they can be used. In our experience, these two factors make the package quite vulnerable to instabilities. For example,
dependencies can easily get in conflict with each other or the files for the pretrained models might be temporarily unavailable or stored in a different location.
To address these issues, we created Docker containers of the pipeline components.

    "A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another."

    -- `Docker website <https://www.docker.com/resources/what-container/>`_

This means that the package can still be used when dependencies are in conflict, break, or the pretrained models cannot be downloaded.

The mexca base package is designed to use the containerized components avoiding the installation of the complex dependencies and downloading the pretrained models.
However, the additional dependencies can also be installed to use the components without containers (see :ref:`Installation`). The first time the pipeline is run with containers,
they will be automatically downloaded from `Docker Hub <https://hub.docker.com/repositories/mexca>`_ which can take some time (they are approx. 17 GB in total, but some layers can be reused).


Downloading a Container
-----------------------

In case the automatic download of a container failed, they can be manualled pull from Docker Hub via:

.. code-block:: console

  docker pull mexca/image-name # e.g., mexca/face-extractor


Running a Container
-------------------

mexca runs the containerized components automatically via the `docker <https://docker-py.readthedocs.io/en/stable/>`_ Python package. The containers can also be run from the CLI via:

.. code-block:: console

  docker run -t -v absolute/path/to/video/folder:/mnt/vol mexca/image-name # add container entrypoint args

Each container has an entrypoint which is a Python CLI and explained in the :ref:`Command Line` section.

*Explanation of the command*:

- The `-t` flag indicates that a specific tag of the Docker image should be run.
- With the `-v` flag, a folder from the host system is mounted onto the container. Here, the absolute path to the folder that should be mounted must be supplied as well as the destination folder in the container (here `/mnt/vol`).
  This allows the container to access the video file to which the pipeline will be applied to and write the output on the host system.
- This is followed by the name of the image and the tag (e.g., `mexca/face-extractor:latest`).
- Append the arguments for the container entrypoint CLI at last (e.g., a file name).


Building an Image
-----------------

The images of the containers can be built from Dockerfiles in the GitHub repository:

.. code-block:: console

  git clone https://github.com/mexca/mexca.git
  cd mexca/docker/component-name # e.g., mexca/docker/face-extractor
  docker build . -t mexca/component-name # e.g., mexca/face-extractor

For details on the build, see the Dockerfiles in the repository.

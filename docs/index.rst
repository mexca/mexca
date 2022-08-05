.. mexca documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mexca's Documentation!
=================================

**Mexca** is an open-source Python package which aims to capture human emotion expressions in videos. It uses pre-trained deep neural networks to identify faces and speakers in videos, and it extracts (i) facial features (e.g., `action units <https://en.wikipedia.org/wiki/Facial_Action_Coding_System#Codes_for_action_units>`_), (ii) acoustic properties of speech (e.g., voice pitch) and (iii) the transcribed speech. The output is a 2-dimensional dataset which contains these features on a fine-grained temporal scale (i.e, frame-by-frame) that can be exported in a .csv format. 

If you would like to learn how to use **mexca**, the best place to start is our `demo <https://github.com/mexca/mexca/tree/main/examples>`_ tutorial.
Contents:

.. toctree::
  :maxdepth: 2

  introduction
  current_status
  installation


==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

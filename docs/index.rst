.. mexca documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Mexca's documentation!
=================================

**Mexca** (**M**ultimodal **E**motional **E**xpression **C**apture **A**msterdam) is an open-source Python package which aims to capture human emotions in video recordings. Mexca uses pre-trained deep neural networks to analyse videos, and it extracts for each recognised individual their (i) subtle facial behaviors (e.g., [action units](https://en.wikipedia.org/wiki/Facial_Action_Coding_System#Codes_for_action_units)), (ii) acoustic properties of speech (e.g., pitch) and (iii) text's sentiment (e.g., whether a given text contains negative, positive, or neutral emotions). The output is a dataset which describes these features on a fine-grained temporal scale (i.e, frame-by-frame) that can be exported in a .csv format. 

If you'd like to learn how to use **Mexca**, the best place to start is our [demo](https://github.com/mexca/mexca/tree/main/examples) tutorial.
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

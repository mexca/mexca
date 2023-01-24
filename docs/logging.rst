Logging
=======

The mexca package includes logging messages using the `logging <https://docs.python.org/3/library/logging.html#>`_ module.
By default, messages on the INFO level and above are displayed in the console when calling ``Pipeline.apply()``. This can be
disabled with setting ``show_progress=False``.

Printing DEBUG messages in the console can be enabled by including the following code at the beginning of a script:

.. code-block:: python

    import logging

    logger = logging.getLogger('mexca')
    logger.setLevel(logging.DEBUG)


Writing DEBUG messges to a log file might be more convenient and can be done via:

.. code-block:: python

    import logging

    logging.basicConfig(filename='mexca.log', encoding='utf-8', level=logging.DEBUG)

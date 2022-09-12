Command Line
============

The **mexca** pipeline can be run via the command line after installing the package:

.. code-block:: console

  mexca-pipeline -f path/to/video_file -o output.json

Note, that if **mexca** has been installed in a virtual environment, the command might need to be executed from a specific location depending on the environment.
For example, when using `venv`:

.. code-block:: console

  ./bin/mexca-pipeline -f path/to/video_file -o output.json

The `mexca-pipeline` command requires two arguments: The path to the video file to be processed and the name of a .json file in which the output is stored (here called output.json as an example). **Note**: Currently, the output file must be .json format.
Furthermore, the script can be run with additional arguments (run the scirpt with the `-h` or `--help` argument):

- `-l`, `--lang`: The language that is transcribed. Currently only Dutch and English are avaiable. Default: English.
- `--face-min`: The minimum number of faces that should be identified. Default: 2.
- `--face-max`: The maximum number of faces that should be identified. Default: None.
- `--speakers`: The number of speakers that should be identified. Default: None.
- `--pitch-low`: The lower bound frequency of the pitch calculation. Default: 75.0.
- `--pitch-high`: The upper bound frequency of the pitch calculation. Default: 300.0
- `--time-step`: The interval between time points at which features are extracted. Only used when video processing is disabled. Default: None.
- `--skip`: Skips every nth video frame. Default: 1.
- `--sublcip`: Process only a part of the video clip. See `moviepy.editor.VideoFileClip <https://moviepy.readthedocs.io/en/latest/ref/VideoClip/VideoClip.html#videofileclip>`_ for details. Default: (0, None).
- `--no-video`: Disables the video processing part of the pipeline. Default: False.
- `--no-audio`: Disables the audio processing part of the pipeline. Default: False.
- `--no-text`: Disables the text processing part of the pipeline. Default: False.

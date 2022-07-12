Introduction
============

The goal of **Mexca** is to apply pre-trained machine learning algorithms to detects emotions 
in three different modalities: video, audio and text in a single pipeline. 

Mexca takes a video as an input, and it returns the following features by frame:

- facial muscle movements (i.e., action units)
- facial landmarks coordinates (x and y coordinates) 
- face labels
- speech segments  
- speaker labels 
- text transcription

The output is a dictionary containing all the above-mentioned features in columns.  


Licensing
---------

Source code and data of mcfly are licensed under the Apache License,
version 2.0.

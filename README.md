## Introduction
This repository is created for the example custom plot scripts for the MindAmp GUI.

## Custom plot in the GUI
To define a proper custom plot script for the GUI, serveral criteria must be fulfilled.

1. A class called ```Custom``` is required to store all the attributes and methods
2. In the ```__init__ ```method of ```Custom```, attribute ```params``` must be included to configure the plot
3. All other one-off setup procedures for the plot should be added to the ```__init__``` method. E.g. Loading a model for signal classification
4. The ```Custom``` class must include the ```custom_func``` method to receive the device signal and return data for the plot

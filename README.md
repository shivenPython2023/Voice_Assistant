# Voice_Assistant
Voice Assistant for Visually Impaired Built on Raspberry Pi 5

*This code works on Windows, Mac, etc.*
*Simply edit where camera footage is pulled from*

Uses a wakeword model for activation. 
The wakeword is 'Suno'.
The wakeword model is saved under 'optimizer2.pt'.

After waking up the model, give an input (via speaking) of either an object (ie. person) or give the command 'all objects'

The code will announce all objects as well as their depths and orientations. 

In order for depth to work, you must have a 2 camera system and calibrate it. Collect images using calibration_images.py and create stereomap.xml by running stereo_calibration.py.

Enjoy!

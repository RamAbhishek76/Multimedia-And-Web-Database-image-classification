# CSE515 Multimedia and Web Databases Project Phase 1

### System requirements
Software and hardware requirements to run this project are:
1. A 64 bit operating system (code has been tested on Windows 10 and Linux Mint 21.2)
2. Python 3.11 or later
3. Mongo compass
4. Mongo Shell (mongosh)

To run this project locally, all the python packages mentioned in the requirements.txt file have to be installed.

### Execution instructions
The steps to run this project locally are:
1. Open Mongo Compass and connect to your local mongo instance.
2. Go the project directory and run ```pip install -r requirements.txt``` to install all the required python packages for this project.
3. Run the python file ```extract_image_features.py``` in the "Code" folder to generate features for all the images in the Caltech101 dataset.
4. Finally run the ```index.py``` file in the "Code" folder to get the user interface and to query the images from the database.


### Functionality of each of the files in the Code/ folder
1. color_moment.py - Has function which can compute the color moment for an image.
2. database_connection.py - Has the defintion of a function which connects to local mongo database.
3. extract_image_features.py - Uses all the feature extraction functions and extracts the features from all the images in the caltech101 dataset.
4. hog.py - Has function which can compute the Histogram of gradients feature for an image.
5. index.py - Has menu based user input to query images from the database
6. output_plotter.py - This file has function definition for a function which allows for the plotting of images once they have been fetched in index.py.
7. query_avgpool, query_color_moment, query_fc, query_hog, query_hog - Contain some test code I used when working on the project.
8. resnet.py - This file has the function definition of the function which can extract layer3, avgpool, fc features for an image.

Submitted by - <br />
Abhinav Gorantla
# face_morphing_demonstrator

# Install

Requirements can be installed inside a virtual environment using the requirement.txt.

pip install -r requirements.txt

Update of the requirements can be conducted with pip freeze.

pip freeze > requirements.txt

# Run

1 - Run "app.py" in your favourite or IDE

2 - Click or type http://0.0.0.0:5000/ in your browser

3- Upload a photo 

5 - See results

# Config

In case certain changes had to be made to values:

	- To change app.py config values. Change values "app_config_utils/appconfig.py"

	- To change landmarks.py config values for width and height desired for image. Change values in "landmarks/lmconfig.py"
	
	- To change face_crop.py config values for left crop, right crop, top crop and bottom crop. Change values in "face_detection/fcconfig.py"

# Computational demand

| Implementation       | Total size          | Requirements| Execution time |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Flask + HTML frontend     | 30 MB | Upload Image to see effect | ≈ 3.5 secs |

# Current limitations

- Landscape images may sometimes be slightly distorted.

- Glasses and beards may affect results.







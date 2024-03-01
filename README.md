# autonomous-vehicle
The project aims to build an autonomous system for vehicles based on computer vision.
Link to the dataset: https://drive.google.com/file/d/10VUs1804rkY6AqN2EEU4RiaHXzENEsGu/view?usp=sharing

# Docker run

To run a code in the docker container please follow the instructions:

1. Download the .zip file containing the dataset 
2. Unzip the file in specified directory <your_dataset_folder>
3. Run the following docker command, mounting the volume to it:

    docker run --rm -v $(PWD)/python/segmentation/<your_dataset_folder>:/app/python/segmentation/dataset -it autonomousvehicle


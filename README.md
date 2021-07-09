# MSc_MAS4061_Dissertation
This repo serves the work on master's thesis

The entire project is designed to run in docker container. 
So, in order to run the project you first need to download
docker (if not already exist) from: https://www.docker.com/products/docker-desktop  

Once downloaded, you need to open a terminal window and 
change the current directory to images folder of this project. Once done,
you then need to run the following commands:
- `docker-compose build` to build the base image. This only need to run once unless 
there's a change in requirements file in which case you need to rerun this.  
- `docker compose up` Once the base image is build you can then run the image.
This will provide a link that you can open in your browser to run jupyter notebooks.  
  
Details of the image used in this project can be found in the docker-compose file under images folder.
# Vegetation-Land-Segmentation-and-Classification
Master Thesis

Goal :

1. Provide a region-independent, vegetation/agriculture zone segmentation model trained on satellite images which can be fine-tuned
with images of another region.
2. Compare and analysis the results of trained model for crop specific during crop cycle in different years. Idea is based on the assumption
crop images share same features during different stages of their cycle every year. 
   


## Setup the environment

* Install all the packages in requirements.txt
* Install pytorch using the [link](https://pytorch.org/get-started/locally/) according to the system.
* Install the [Detectron2 framework](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)


## Preprocessing

* Satellite Images
    * Sentinel-2 dataset downloaded for 4 time intervals 
      * Jan-March  (Less than 2 percent cloud)
      * Apr- Jun   (Less than 2 percent cloud)
      * July - Sep  (Less than 2 percent cloud)
      * Oct- Dec   (Less than 8 percent cloud)

* Shape Files
    * Shape files are provided by [FVM](https://www.geodata-info.dk/srv/eng/catalog.search#/home)
    * Based on the knowledge of crop cycle and crops grown in denmark, thesis focused on 7 types of crops. More information 
    about the categorisation can be found [here](https://docs.google.com/drawings/d/1oyH4NqZqckdJBXbudCC-5BtxOSA5LPmg3pojAGm8RP4/edit?usp=sharing)
  * Crop reclassification labels can be found in config/classes.py
* 5 versions of dataset is created for training, four multi-class classification and single-class classification. Oct-dec is omitted 
due to high cloud percentage.
  
* In order to generate training and validation data and preprocessing, follow Preprocessing.ipynb or run generate_data.py.
You may need to change configuration file config/config.py
  
## Detectron2
* For implementation of Mask R-CNN, Facebook's Detectron2 framework is used. It is modified to serve the thesis's training.
* For Mask-RCNN, train.py is used to train on DFKI GPU clusters. You have to change the config/config.py according to your systems path
* For testing, a separate test.py script can be used.

## FCIS
* For implementation of FCIS, MxNet framework needs to be setup. Please see FCIS directory.


  

  


        

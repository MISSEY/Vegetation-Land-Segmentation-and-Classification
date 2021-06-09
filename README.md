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
      
        **Summmer and Winter crop categorisation**

        | Season      | crop |
        | ----------- | ----------- |
        | Winter      | barley_w, wheat_w, rapeseed_w, rye_w       |
        | Summer/Spring   | barley_s,wheat_s,oats_s        |
      **1. Jan-Mar**

        | crop      | Crop_nr |   cycle_stage     |
        | ----------- | ----------- | ----------- |
        |  barley_w     |   10     | mid |
        | wheat_w   | 11,13       | mid |
        | rye_w      | 14,15      | mid |
        | rapeseed_w      | 22     | mid |

        **2. Apr-Jun**
        
        | crop      | Crop_nr |   cycle_stage     |
        | ----------- | ----------- | ----------- |
        |  barley_w     |   10     | mid |
        | wheat_w   | 11,13       | mid |
        | rye_w      | 14,15      | mid |
        | rapeseed_w      | 22     | mid |
        | barley_s   | 01      | planting and mid  |
        | wheat_s      | 02,06     | planting and mid |
        | oats_s      | 03   | planting and mid |

        **3. Jul-Sep**
        
        | crop      | Crop_nr |   cycle_stage     |
        | ----------- | ----------- | ----------- |
        |  barley_w     |   10     | mid and harvest |
        | wheat_w   | 11,13       | mid and harvesst|
        | rye_w      | 14,15      | harvest |
        | rapeseed_w      | 22     | harvest |
        | barley_s   | 01     | mid and harvest |
        | wheat_s      | 02,06     | mid and harvest |
        | oats_s      | 03   | mid and harvest  |

        **4. Oct-Dec**
        
        | crop      | Crop_nr |   cycle_stage     |
        | ----------- | ----------- | ----------- |
        |  barley_w     |   10     | planting |
        | wheat_w   | 11,13       | planting|
        | rye_w      | 14,15      | planting |
        | rapeseed_w      | 22     | planting |
        | barley_s   | 01     |  harvest |
        | wheat_s      | 02,06     | harvest |
    
* 5 versions of dataset is created for training, 4 for 4 stages of crop cycle and 1 additional for classification of summer and 
winter crops for whole year. 
  


        

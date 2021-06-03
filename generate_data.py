"""
Only focused on Vegetation land segmentation, please do not use as generalised purposes. As global variables are
hard coded which are file path, that can be differ for some other applications

Generate data from downloaded raster image
This file only responsible for automation of croping, slipting, dataset creation .
for more description please follow Preprocessing.ipynb

"""


import os
import rasterio
import shapely
from tqdm import tqdm
from shutil import copyfile
import rasterio.mask
from common import coco_utils
from data_op import rasterfile as rf
from data_op import shapefile
from config import settings as st
from common import dictionary_utils
import data_op.shape_opr as so
from config import config as cfg

# Global variables
denmark_shape_directory = cfg.denmark_shape_directory
_preprocessed_denamark_shape_files_ = cfg._preprocessed_denamark_shape_files_

# File path where vector shape files for each categories are saved after filtering
# Please see Preprocessing.ipynb for categorisation.

reformatted_file_path = os.path.join(st.raw_shape_directory, 'Reformatted', denmark_shape_directory)
processed_file_path = os.path.join(st.data_directory, _preprocessed_denamark_shape_files_, denmark_shape_directory)

year = cfg.generate_year
year_processed = year + '_processed'

image_size = cfg.data_generation_image_size

denmark_tif = os.path.join(st.data_directory, cfg._tif_, year)

def create_directory(path):
    """
    Create directories if not found
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Creation of the directory %s Success" % path)


def preprocess_shape_files():
    """
    Preprocess each shape files for different categories and save it to processed file directory for further processing
    :return: None
    """
    create_directory(processed_file_path)

    # For year shape files
    for file in tqdm(os.listdir(reformatted_file_path)):
        file_type, filename = file.split('.')[1], file.split('.')[0]
        if file_type == 'shp':
            # read the shape fil;es
            shape_ex = shapefile.Shape_Extractor(shape_file=os.path.join(reformatted_file_path, file))
            denmark_veg = shape_ex.import_shape()

            # explode all multipolygon to polygons
            denmark_veg = shape_ex.explode(df=denmark_veg)
            denmark_veg = shape_ex.buffer_zero(denmark_veg)
            denmark_veg = denmark_veg.reset_index(drop=True)
            denmark_veg = denmark_veg.assign(fid=lambda _df: range(0, len(_df.index)))

            # save it to directory
            denmark_veg.to_file(os.path.join(processed_file_path, file))


def clip_shape_on_raster_bounds():
    """
    Clip each shape files on respective raster data and save it to directory
    :return: None
    """
    # create directory for yearly processed shape files

    path = os.path.join(st.data_directory, year_processed)
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Creation of the directory %s Success" % path)

    # For each tif crop each category shape files and save in the processed folder

    for directory in tqdm(os.listdir(denmark_tif)):
        shape_ex = shapefile.Shape_Extractor(shape_file=os.path.join(processed_file_path, str(directory) + '.shp'))
        denmark_veg = shape_ex.import_shape()

        # crs used while extracting the raster image
        denmark_veg = denmark_veg.to_crs(cfg.crs)

        # For each raster (4-5 categories)
        for file in tqdm(os.listdir(os.path.join(denmark_tif, directory))):
            # load raster and extract bounds for cliping
            with rasterio.open(os.path.join(denmark_tif, directory, file)) as src:
                raster_meta = src.meta
                raster_bounds = src.bounds

            filename = file.split('.')[0]

            # clip the shape files on raster bounds
            df = shape_ex.clip(df=denmark_veg, clip_poly=shapely.geometry.box(*raster_bounds), keep_biggest_poly_=True)

            df = df.assign(area_sqm=df.geometry.area)

            # save it in processed directory for each categories
            path = os.path.join(st.data_directory, year_processed, 'v_' + directory, cfg._version_processed_shape_files)
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Creation of the directory %s Success" % path)
            df.to_file(os.path.join(path, str(filename) + '.shp'))


def crop_vector_into_chips():
    """
    Crop vectors in chip defined image sizes and later on crop raster image
    :return: None
    """

    # for each tif
    for directory in tqdm(os.listdir(denmark_tif)):

        # count for generation of unique image
        count = 0

        # track the image_id_count
        file_name_image_id_count_dictionary = {}
        final_chip_dfs = {}
        version_path = os.path.join(st.data_directory, year_processed, 'v_' + directory)

        for file in tqdm(os.listdir(os.path.join(denmark_tif, directory))):

            # read raster image
            with rasterio.open(os.path.join(denmark_tif, directory, file)) as src:
                raster_meta = src.meta
                raster_bounds = src.bounds
            filename = file.split('.')[0]

            # read corresponding vector
            shape_ex = shapefile.Shape_Extractor(
                shape_file=os.path.join(version_path, cfg._version_processed_shape_files, str(filename) + '.shp'))
            denmark_veg = shape_ex.import_shape()
            prev = count

            chip_name_prefix = 'COCO_train' + year + '_' + directory + '_000000'

            # crop the vector on raster height and width of defing image sixe and save the information into pickle
            chip_dfs, count = so.crop_vector_in_chips(df=denmark_veg,
                                                      raster_width=raster_meta['width'],
                                                      raster_height=raster_meta['height'],
                                                      raster_transform=raster_meta['transform'],
                                                      chip_width=image_size,
                                                      chip_height=image_size,
                                                      count=count,
                                                      chipname=chip_name_prefix,
                                                      skip_partial_chips=True)

            # create directory for each version to save chip info which later used on croping raster image
            path = os.path.join(version_path, str(image_size), 'crop_chip_info')
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Creation of the directory %s Success" % path)
            # save the dictionary in pickle,
            chip_path = os.path.join(path, str(filename) + '.pickle')
            final_chip_dfs.update(chip_dfs)
            file_name_image_id_count_dictionary[filename] = list(range(prev, count))
            dictionary_utils.new_pickle(chip_path, chip_dfs)

        # save final crop chip merging for each raster for the respective vector
        path = os.path.join(version_path, str(image_size), 'final_crop_chip_info')
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Creation of the directory %s Success" % path)

        # save in pickle
        dictionary_utils.new_pickle(os.path.join(path, 'chip_dfs.pickle'), final_chip_dfs)
        dictionary_utils.new_pickle(os.path.join(path, 'file_name_image_id_count_dictionary.pickle'),
                                    file_name_image_id_count_dictionary)


def crop_raster_image():
    """
    Crop raster as did for vector , load the saved chip info and crop raster on defined image size and save it to
    directory
    :return:
    """
    for directory in tqdm(os.listdir(denmark_tif)):
        version_path = os.path.join(st.data_directory, year_processed, 'v_' + directory)
        pickle_path = os.path.join(version_path, str(image_size), 'final_crop_chip_info')

        # load the saved chip infos
        file_name_image_id_count_dictionary = dictionary_utils.load_pickle(
            os.path.join(pickle_path, 'file_name_image_id_count_dictionary.pickle')
        )
        final_chip_dfs = dictionary_utils.load_pickle(
            os.path.join(pickle_path, 'chip_dfs.pickle')
        )
        for file in tqdm(os.listdir(os.path.join(denmark_tif, directory))):
            filename = file.split('.')[0]
            chip_dfsss = file_name_image_id_count_dictionary[filename]
            chip_windows = {}
            raster_image_max = 2200  # Value used while extracting the sentinel-2 image
            path = os.path.join(version_path, str(image_size), cfg._version_crop_images_)
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Creation of the directory %s Success" % path)

            chip_name_prefix = 'COCO_train' + year + '_' + directory + '_000000'

            # for each chip crop the raster image based on bounds and save it to directory
            for chip_no in chip_dfsss:
                chip_name = chip_name_prefix+f'{100000 + chip_no}'
                chip_windows.update({chip_name: final_chip_dfs[chip_name]['chip_window']})

            # cut chips
            stats = rf.cut_chip_images(inpath_raster=os.path.join(denmark_tif, directory, file),
                                       outpath_chipfolder=path,
                                       chip_names=chip_windows.keys(),
                                       chip_windows=chip_windows.values(),
                                       raster_image_range=raster_image_max)

def save_vectors_in_coco_annotations():
    """
    Save vectors in coco dataset format
    :return: None
    """

    for directory in tqdm(os.listdir(denmark_tif)):
        version_path = os.path.join(st.data_directory, year_processed, 'v_' + directory)
        pickle_path = os.path.join(version_path, str(image_size), 'final_crop_chip_info', 'chip_dfs.pickle')

        # saved shape file
        file = directory + '.shp'
        shape_ex = shapefile.Shape_Extractor(shape_file=os.path.join(processed_file_path, file))
        denmark_veg = shape_ex.import_shape()

        final_chip_dfs = dictionary_utils.load_pickle(pickle_path)

        # split into training and validation sets
        train_chip_dfs, val_chip_dfs = coco_utils.train_test_split(final_chip_dfs, test_size=0.2, seed=1)

        coco_train = coco_utils.format_coco(train_chip_dfs, image_size, image_size, denmark_veg)
        coco_val = coco_utils.format_coco(val_chip_dfs, image_size, image_size, denmark_veg)

        path_train = os.path.join(version_path, str(image_size), cfg._version_train_)
        path_val = os.path.join(version_path, str(image_size), cfg._version_validation_)

        try:
            os.makedirs(os.path.join(path_train, 'images'))
            os.makedirs(os.path.join(path_train, 'annotation'))
            os.makedirs(os.path.join(path_val, 'images'))
            os.makedirs(os.path.join(path_val, 'annotation'))

        except OSError:
            print("Creation of the directory {0} and {1}  failed".format(path_train, path_val))
        else:
            print("Creation of the directory {0} and {1} Success".format(path_train, path_val))

        dictionary_utils.new_json(outpath=os.path.join(path_train, 'annotation/train'+year+'.json'), data=coco_train)
        dictionary_utils.new_json(outpath=os.path.join(path_val, 'annotation/val'+year+'.json'), data=coco_val)

        ## Split the images in train and validation
        # Split the cropped images into train and validation with the help of train_chip_dfs and val_chip_dfs

        train_images, validation_images = list(train_chip_dfs.keys()), list(val_chip_dfs.keys())

        cropped_path = os.path.join(version_path, str(image_size), cfg._version_crop_images_)
        for image in validation_images:
            img = image.replace('val', 'train')
            copyfile(os.path.join(cropped_path, str(img) + '.jpg'),
                     os.path.join(path_val, 'images/' + str(image) + '.jpg'))

        for image in train_images:
            copyfile(os.path.join(cropped_path, str(image) + '.jpg'),
                     os.path.join(path_train, 'images/' + str(image) + '.jpg'))


if __name__ == '__main__':
    # 1. preprocess
    # print("Preprocessing")
    # preprocess_shape_files()
    # print("Finish Preprocessing")
    #
    # print("Clipping on raster bounds")
    # # 2. clip shape on raster bounds
    # clip_shape_on_raster_bounds()
    # print("finish Clipping on raster bounds")

    print("Vector crop")
    # 3. crop the vectors of defined image size
    crop_vector_into_chips()
    print("Finish Vector crop")

    print("Raster image crop")
    # 4. crop raster of defined image size
    crop_raster_image()
    print("Finish Raster image crop")

    print("Annotations")
    # 5. split validation and training set and save into coco data format
    save_vectors_in_coco_annotations()
    print("Finish annotations")

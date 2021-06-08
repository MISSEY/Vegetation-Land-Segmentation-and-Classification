from typing import Union, Tuple, List, Dict
from pathlib import Path
import random
import itertools


import base64
import IPython
import json
import numpy as np
import os
import random
import requests
from io import BytesIO
from math import trunc
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import common.dictionary_utils as dic
import common.coco_utils
from matplotlib import pyplot as plt

from descartes import PolygonPatch
from PIL import Image as pilimage
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon

def train_test_split(chip_dfs: Dict, test_size=0.2, seed=1) -> Tuple[Dict, Dict]:
    """Split chips into training and test set.

    Args:
        chip_dfs: Dictionary containing key (filename of the chip) value (dataframe with
            geometries for that chip) pairs.
        test_size: Relative number of chips to be put in the test dataset. 1-test_size is the size of the
        training data set.
    """
    chips_list = list(chip_dfs.keys())
    random.seed(seed)
    random.shuffle(chips_list)
    split_idx = round(len(chips_list) * test_size)
    train_split = chips_list[split_idx:]
    val_split = chips_list[:split_idx]

    train_chip_dfs = {k: chip_dfs[k] for k in sorted(train_split)}
    val_chip_dfs = {k.replace('train', 'val'): chip_dfs[k] for k in sorted(val_split)}

    return train_chip_dfs, val_chip_dfs

def format_coco(chip_dfs: Dict, chip_width: int, chip_height: int,df):
    """Format train and test chip geometries to COCO json format.

    Args:
        chip_dfs: Dictionary containing key (filename of the chip) value (dataframe with
            geometries for that chip) pairs.
        chip_width: width of the chip in pixel size.
        chip_height: height of the chip in pixel size.
        df: original geopandas dataframe which also include in chip_dfs

    COCOjson example structure and instructions below. For more detailed information on building a COCO
        dataset see http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

    cocojson = {
        "info": {...},
        "licenses": [...],
        "categories": [{"supercategory": "person","id": 1,"name": "person"},
                       {"supercategory": "vehicle","id": 2,"name": "bicycle"},
                       ...],
        "images":  [{"file_name": "000000289343.jpg", "height": 427, "width": 640, "id": 397133},
                    {"file_name": "000000037777.jpg", "height": 230, "width": 352, "id": 37777},
                    ...],
        "annotations": [{"segmentation": [[510.66,423.01,...,510.45,423.01]], "area": 702.10, "iscrowd": 0,
                         "image_id": 289343, "bbox": [473.07,395.93,38.65,28.67], "category_id": 18, "id": 1768},
                        {"segmentation": [[340.32,758.01,...,134.25,875.01]], "area": 342.08, "iscrowd": 0,
                         "image_id": 289343, "bbox": [473.07,395.93,38.65,28.67], "category_id": 18, "id": 1768},
                         ...]
        }

    - "id" in "categories" has to match "category_id" in "annotations".
    - "id" in "images" has to match "image_id" in "annotations".
    - "segmentation" in "annotations" is encoded in Run-Length-Encoding (except for crowd region (iscrowd=1)).
    - "id" in "annotations has to be unique for each geometry, so 4370 geometries in 1000 chips > 4370 uniques
       geometry ids. However, does not have to be unique between coco train and validation set.
    - "file_name" in "images" does officially not have to match the "image_id" in "annotations" but is strongly
       recommended.
    """

    dict_id_category = get_categories(df)

    cocojson = {
        "info": {},
        "licenses": [],
    }
    for key,value in dict_id_category.items():
        categories = {
            'supercategory': 'Agriculture',
            'id': key,
            'name':value
        }
        cocojson.setdefault('categories',[]).append(categories)

    annotation_id = 1
    none_values = 0
    for chip_name in chip_dfs.keys():
        chip__name_split = chip_name.split('_')
        if 'train' in chip_name:
            chip_id = int(chip__name_split[-1])
        elif 'val' in chip_name:
            chip_id = int(chip__name_split[-1])

        image = {"file_name": f'{chip_name}.jpg',
                  "id": int(chip_id),
                  "height": chip_height,
                  "width": chip_width}
        cocojson.setdefault('images', []).append(image)

        for _, row in chip_dfs[chip_name]['chip_df'].iterrows():
            # Convert geometry to COCO segmentation format:
            # From shapely POLYGON ((x y, x1 y2, ..)) to COCO [[x, y, x1, y1, ..]].
            # The annotations were encoded by RLE, except for crowd region (iscrowd=1)
            coco_xy = list(itertools.chain.from_iterable((x, y) for x, y in zip(*row.geometry.exterior.coords.xy)))
            coco_xy = [round(coords, 2) for coords in coco_xy]
            # Add COCO bbox in format [minx, miny, width, height]
            bounds = row.geometry.bounds  # COCO bbox
            coco_bbox = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]
            coco_bbox = [round(coords, 2) for coords in coco_bbox]
            if(row['r_code'] is None):
                none_values = none_values + 1
                print(none_values)
            else:
                annotation = {"id": annotation_id,
                               "image_id": int(chip_id),
                               "category_id": int(row['r_code']),  # with multiple classes use "category_id" : row.reclass_id
                               "mycategory_name": row['r_classes'],
                               "bbox": coco_bbox,
                               "area": row.geometry.area,
                               "segmentation": [coco_xy]}
                cocojson.setdefault('annotations', []).append(annotation)

                annotation_id += 1


    return cocojson

def get_categories(df):
    """

    :param df:
    :return:
    """
    dict_id_category = {}
    for i in range(len(df.groupby('r_code').groups)):
        dict_id_category[int(df.groupby('r_code').get_group(i + 1).iloc[0]['r_code'])] = str(df.groupby('r_code').get_group(i + 1).iloc[0]['r_classes'])

    return dict_id_category

def coco_to_shapely(inpath_json: Union[Path, str],
                    categories: List[int] = None) -> Dict:
    """Transforms COCO annotations to shapely geometry format.

    Args:
        inpath_json: Input filepath coco json file.
        categories: Categories will filter to specific categories and images that contain at least one
        annotation of that category.

    Returns:
        Dictionary of image key and shapely Multipolygon.
    """

    data = dic.load_json(inpath_json)
    if categories is not None:
        # Get image ids/file names that contain at least one annotation of the selected categories.
        image_ids = sorted(list(set([x['image_id'] for x in data['annotations'] if x['category_id'] in categories])))
    else:
        image_ids = sorted(list(set([x['image_id'] for x in data['annotations']])))
    file_names = [x['file_name'] for x in data['images'] if x['id'] in image_ids]

    # Extract selected annotations per image.
    extracted_geometries = {}
    for image_id, file_name in zip(image_ids, file_names):
        annotations = [x for x in data['annotations'] if x['image_id'] == image_id]
        if categories is not None:
            annotations = [x for x in annotations if x['category_id'] in categories]

        segments = [segment['segmentation'][0] for segment in annotations]  # format [x,y,x1,y1,...]

        # Create shapely Multipolygons from COCO format polygons.
        mp = MultiPolygon([Polygon(np.array(segment).reshape((int(len(segment) / 2), 2))) for segment in segments])
        extracted_geometries[str(file_name)] = mp

    return extracted_geometries

def plot_coco(inpath_json, inpath_image_folder, start=0, end=2):
    """Plot COCO annotations and image chips"""
    extracted = common.coco_utils.coco_to_shapely(inpath_json)

    for key in sorted(extracted.keys())[start:end]:
        print(key)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        path = os.path.join(inpath_image_folder,key)
        img = np.asarray(pilimage.open(path))
        plt.imshow(img, interpolation='none')

        mp = extracted[key]
        patches = [PolygonPatch(p, ec='r', fill=False, alpha=1, lw=0.7, zorder=1) for p in mp]
        plt.gca().add_collection(PatchCollection(patches, match_original=True))
        plt.show()
class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink', 'gold',
                       'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                       'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                       'magenta', 'sienna', 'maroon']

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        self.process_info()
        # self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))

        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')

    def display_licenses(self):
        print('Licenses:')
        print('=========')

        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(req, str(req_type)))
            print('')
        print('')

    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(cat_id, self.categories[cat_id]['name']))
            print('')

    def display_image(self, image_id, show_polys=True, show_bbox=True, show_labels=True, show_crowds=True,
                      use_url=False):
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))

        # Print the image info
        image = self.images[image_id]
        for key, val in image.items():
            print('  {}: {}'.format(key, val))

        # Open the image
        if use_url:
            image_path = image['coco_url']
            response = requests.get(image_path)
            image = PILImage.open(BytesIO(response.content))

        else:
            image_path = os.path.join(self.image_dir, image['file_name'])
            image = PILImage.open(image_path)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = "data:image/png;base64, " + base64.b64encode(buffered.getvalue()).decode()

        # Calculate the size and adjusted display size
        max_width = 900
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        labels = {}
        print('  segmentations ({}):'.format(len(self.segmentations[image_id])))
        for i, segm in enumerate(self.segmentations[image_id]):
            polygons_list = []
            if segm['iscrowd'] != 0:
                # Gotta decode the RLE
                px = 0
                x, y = 0, 0
                rle_list = []
                for j, counts in enumerate(segm['segmentation']['counts']):
                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Need to draw on these pixels, since we are drawing in vector form,
                        # we need to draw horizontal lines on the image
                        x_start = trunc(trunc(px / image_height) * adjusted_ratio)
                        y_start = trunc(px % image_height * adjusted_ratio)
                        px += counts
                        x_end = trunc(trunc(px / image_height) * adjusted_ratio)
                        y_end = trunc(px % image_height * adjusted_ratio)
                        if x_end == x_start:
                            # This is only on one line
                            rle_list.append({'x': x_start, 'y': y_start, 'width': 1, 'height': (y_end - y_start)})
                        if x_end > x_start:
                            # This spans more than one line
                            # Insert top line first
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (image_height - y_start)})

                            # Insert middle lines if needed
                            lines_spanned = x_end - x_start + 1  # total number of lines spanned
                            full_lines_to_insert = lines_spanned - 2
                            if full_lines_to_insert > 0:
                                full_lines_to_insert = trunc(full_lines_to_insert * adjusted_ratio)
                                rle_list.append(
                                    {'x': (x_start + 1), 'y': 0, 'width': full_lines_to_insert, 'height': image_height})

                            # Insert bottom line
                            rle_list.append({'x': x_end, 'y': 0, 'width': 1, 'height': y_end})
                if len(rle_list) > 0:
                    rle_regions[segm['id']] = rle_list
            else:
                # Add the polygon segmentation
                for segmentation_points in segm['segmentation']:
                    segmentation_points = np.multiply(segmentation_points, adjusted_ratio).astype(int)
                    polygons_list.append(str(segmentation_points).lstrip('[').rstrip(']'))

            polygons[segm['id']] = polygons_list

            if i < len(self.colors):
                poly_colors[segm['id']] = self.colors[i]
            else:
                poly_colors[segm['id']] = 'white'

            bbox = segm['bbox']
            bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                           bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3],
                           bbox[0], bbox[1]]
            bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
            bbox_polygons[segm['id']] = str(bbox_points).lstrip('[').rstrip(']')

            labels[segm['id']] = (self.categories[segm['category_id']]['name'], (bbox_points[0], bbox_points[1] - 4))

            # Print details
            #print('    {}:{}:{}'.format(segm['id'], poly_colors[segm['id']], self.categories[segm['category_id']]))

        # Draw segmentation polygons on image
        html = '<div class="container" style="position:relative;">'
        html += '<img src="{}" style="position:relative;top:0px;left:0px;width:{}px;">'.format(img_str, adjusted_width)
        html += '<div class="svgclass"><svg width="{}" height="{}">'.format(adjusted_width, adjusted_height)

        if show_polys:
            for seg_id, points_list in polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for points in points_list:
                    html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5" />'.format(
                        points, fill_color, stroke_color)

        if show_crowds:
            for seg_id, rect_list in rle_regions.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for rect_def in rect_list:
                    x, y = rect_def['x'], rect_def['y']
                    w, h = rect_def['width'], rect_def['height']
                    html += '<rect x="{}" y="{}" width="{}" height="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5; stroke-opacity:0.5" />'.format(
                        x, y, w, h, fill_color, stroke_color)

        if show_bbox:
            for seg_id, points in bbox_polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0" />'.format(
                    points, fill_color, stroke_color)

        if show_labels:
            for seg_id, label in labels.items():
                color = poly_colors[seg_id]
                html += '<text x="{}" y="{}" style="fill:{}; font-size: 12pt;">{}</text>'.format(label[1][0],
                                                                                                 label[1][1], color,
                                                                                                 label[0])

        html += '</svg></div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass { position:absolute; top:0px; left:0px;}'
        html += '</style>'
        return html

    def process_info(self):
        self.info = self.coco['info']

    def process_licenses(self):
        self.licenses = self.coco['licenses']

    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}  # Create a new set with the category id
            else:
                self.super_categories[super_category] |= {cat_id}  # Add category id to the set

    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def return_segmentation(self,image_id):
        print('  segmentations ({}):'.format(len(self.segmentations[image_id])))
        segmentations = []
        for i, segm in enumerate(self.segmentations[image_id]):
            segmentations.append(segm)
        return segmentations



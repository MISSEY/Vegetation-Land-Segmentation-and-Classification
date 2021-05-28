from tqdm import tqdm

from shapely.geometry import Polygon

import shapely
from typing import Union, Dict

from geopandas import GeoDataFrame as GDF
import geopandas as gpd
from shapely.geometry import Polygon,MultiPolygon
from shapely.geometry import LineString, shape

import rasterio.crs
import data_op.rasterfile


def to_pixelcoords(ingeo,
                   reference_bounds: Union[rasterio.coords.BoundingBox, tuple],
                   scale: bool = False,
                   nrows: int = None,
                   ncols: int = None
                   ) -> Union[Polygon, GDF]:
    """Converts projected polygon coordinates to pixel coordinates of an image array.

    Subtracts point of origin, scales to pixelcoordinates.

    Input:
        ingeo: input geodataframe or shapely Polygon.
        reference_bounds:  Bounding box object or tuple of reference (e.g. image chip) in format (left, bottom,
            right, top)
        scale: Scale the polygons to the image size/resolution. Requires image array nrows and ncols parameters.
        nrows: image array nrows, required for scale.
        ncols: image array ncols, required for scale.

    Returns:
        Result polygon or geodataframe, same type as input.
    """

    def _to_pixelcoords(poly, reference_bounds, scale, nrows, ncols):
        try:
            minx, miny, maxx, maxy = reference_bounds
            w_poly, h_poly = (maxx - minx, maxy - miny)
        except (TypeError, ValueError):
            raise Exception(
                f'reference_bounds argument is of type {type(reference_bounds)}, needs to be a tuple or rasterio '
                f'bounding box '
                f'instance. Can be delineated from transform, nrows, ncols via rasterio.transform.reference_bounds')

        # Subtract point of origin of image bbox.

        # For line street
        if LineString == type(shape(poly)):
            x_coords, y_coords = poly.coords.xy
            p_origin = shapely.geometry.LineString([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])
        else:
            x_coords, y_coords = poly.exterior.coords.xy  # For polygon
            p_origin = shapely.geometry.Polygon([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])

        if scale is False:
            return p_origin
        elif scale is True:
            if ncols is None or nrows is None:
                raise ValueError('ncols and nrows required for scale')
            x_scaler = ncols / w_poly
            y_scaler = nrows / h_poly
            return shapely.affinity.scale(p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    if isinstance(ingeo, Polygon):
        return _to_pixelcoords(poly=ingeo, reference_bounds=reference_bounds, scale=scale, nrows=nrows, ncols=ncols)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _to_pixelcoords(poly=_p, reference_bounds=reference_bounds,
                                                                         scale=scale, nrows=nrows, ncols=ncols))
        return ingeo


def invert_y_axis(ingeo: Union[Polygon, GDF],
                  reference_height: int
                  ) -> Union[Polygon, GDF]:
    """Invert y-axis of polygon or geodataframe geometries in reference to a bounding box e.g. of an image chip.

    Usage e.g. for COCOJson format.

    Args:
        ingeo: Input Polygon or geodataframe.
        reference_height: Height (in coordinates or rows) of reference object (polygon or image, e.g. image chip.

    Returns:
        Result polygon or geodataframe, same type as input.
    """

    def _invert_y_axis(poly: Polygon = ingeo, reference_height=reference_height):
        if LineString == type(shape(poly)):
            x_coords, y_coords = poly.coords.xy
            p_inverted_y_axis = shapely.geometry.LineString(
                [[x, reference_height - y] for x, y in zip(x_coords, y_coords)])
        else:
            x_coords, y_coords = poly.exterior.coords.xy
            p_inverted_y_axis = shapely.geometry.Polygon(
                [[x, reference_height - y] for x, y in zip(x_coords, y_coords)])
        return p_inverted_y_axis

    if isinstance(ingeo, Polygon):
        return _invert_y_axis(poly=ingeo, reference_height=reference_height)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(
            lambda _p: _invert_y_axis(poly=_p, reference_height=reference_height))
        return ingeo


def crop_vector_in_chips(df, raster_width, raster_height, raster_transform, chip_width, chip_height, count, chipname,
                         skip_partial_chips):
    """

    :param df:
    :param raster_width:
    :param raster_height:
    :param raster_transform:
    :param chip_width:
    :param chip_height:
    :param skip_partial_chips:
    :return:
    """

    from data_op import shapefile
    shape_ex = shapefile.Shape_Extractor()

    generator_window_bounds = data_op.rasterfile.get_chip_windows(raster_width=raster_width,
                                                                  raster_height=raster_height,
                                                                  raster_transform=raster_transform,
                                                                  chip_width=chip_width,
                                                                  chip_height=chip_height,
                                                                  skip_partial_chips=True)

    all_chip_dfs = {}
    for i, (chip_window, chip_transform, chip_poly) in enumerate(tqdm(generator_window_bounds)):
        chip_df = shape_ex.clip(df, clip_poly=chip_poly,keep_biggest_poly_ = True)
        chip_df = to_pixelcoords(chip_df, reference_bounds=chip_poly.bounds, scale=True, ncols=chip_width,
                                 nrows=chip_height)
        chip_df = invert_y_axis(chip_df, reference_height=chip_height)

        chip_name = chipname+f'{100000 + count}'  # _{clip_minX}_{clip_minY}_{clip_maxX}_{clip_maxY}'

        all_chip_dfs[chip_name] = {'chip_df': chip_df,
                                   'chip_window': chip_window,
                                   'chip_transform': chip_transform,
                                   'chip_poly': chip_poly}
        count = count + 1
    return all_chip_dfs,count

def convert_line_poly(df,xlim,ylim,crs):
    """

    :param df: geopandas dataframe
    :param xlim: x breadth for creating polygon
    :param ylim: y breadth for creating polygon
    :return: geopandas with polygon
    """

    dictionary = {'geometry':[]}
    columns = df.columns
    for column in columns:
        if(column != 'geometry'):
            dictionary[column] = []
    for idx,row in tqdm(df.iterrows()):
        line = row.geometry
        xy = line.xy
        polygon_x = []
        polygon_y = []
        for x_,y_ in zip(xy[0],xy[1]):
            x_plus = x_ + xlim
            y_plus = y_ + ylim
            polygon_y.append((x_,y_plus))
            polygon_x.append((x_plus,y_))
        for x_,y_ in zip(list(reversed(xy[0])),list(reversed(xy[1]))):
            y_minus = y_ - ylim
            x_minus = x_ - xlim
            polygon_y.append((x_,y_minus))
            polygon_x.append((x_minus,y_))
        poly_x = Polygon(polygon_x)
        poly_y = Polygon(polygon_y)
        #multi = MultiPolygon([poly_x,poly_y])
        #geometry.append(multi)
        for i in range(2):
            if(i==0):
                dictionary['geometry'].append(poly_x)
                for column in columns:
                    if(column != 'geometry'):
                        dictionary[column].append(row[column])
            elif(i==1):
                dictionary['geometry'].append(poly_y)
                for column in columns:
                    if (column != 'geometry'):
                        dictionary[column].append(row[column])
    gdf = gpd.GeoDataFrame(dictionary, crs=crs)

    return gdf
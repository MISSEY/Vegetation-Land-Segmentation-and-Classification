"""
    Extracting shape_files for regions
"""

# imports

from config import settings as st

import geopandas as gpd
from tqdm import tqdm
import numpy as np
from geopandas import GeoDataFrame as GDF
from shapely.geometry import Polygon

class Shape_Extractor:

    def __init__(self, shape_file=None,raster_file=None):
        self.shape_file = shape_file
        self.raster_file = raster_file

    def import_shape(self):
        """
        Import shape data
        :return: geopandas dataframe
        """
        if self.shape_file is not None:
            shapes = gpd.read_file(self.shape_file)
            return shapes

    def buffer_zero(self,ingeo):
        """Make invalid polygons (due to self-intersection) valid by buffering with 0."""
        if isinstance(ingeo, Polygon):
            if ingeo.is_valid is False:
                return ingeo.buffer(0)
            else:
                return ingeo
        elif isinstance(ingeo, GDF):
            if False in ingeo.geometry.is_valid.unique():
                ingeo.geometry = ingeo.geometry.apply(lambda _p: _p.buffer(0))
                return ingeo
            else:
                return ingeo

    def filter_shape_data(self, df, column_names):
        """

            Filter out the data based on columns and types
            :param df: Geopandas dataframe
            :param column_names: name of columns to remove
            :return: geopanda dataframe

        """

        # rectifying for geometries that cross themselves.
        if False in df.geometry.is_simple.unique():
            df = df[df.layer == 0]

        # Check whether the geometry are valid or not
        if False in df.geometry.is_valid.unique():
            df = df[df.geometry.is_valid]

        df = df.drop(column_names, axis=1)

        return df

    def explode_mp(self,df):
        """Explode all multi-polygon geometries in a geodataframe into individual polygon geometries.

        Adds exploded polygons as rows at the end of the geodataframe and resets its index.
        """
        outdf = df[df.geom_type == 'Polygon']

        df_mp = df[df.geom_type == 'MultiPolygon']
        for idx, row in tqdm(df_mp.iterrows()):
            df_temp = gpd.GeoDataFrame(columns=df_mp.columns)
            df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
            for i in range(len(row.geometry)):
                df_temp.loc[i, 'geometry'] = row.geometry[i]
            outdf = outdf.append(df_temp, ignore_index=True)

        outdf.reset_index(drop=True, inplace=True)
        return outdf

    def explode_all_ms(self,df):
        """
        Explode all mulit-line string into line-string
        Adds exploded multi-line string as rows at the end of the geodataframe and resets its index.

        :param df: Geopandas dataframe
        :return: geopandas
        """

        outdf = df[df.geom_type == 'LineString']
        df_ms = df[df.geom_type == 'MultiLineString']
        for idx, row in tqdm(df_ms.iterrows()):
            df_temp = gpd.GeoDataFrame(columns=df_ms.columns)
            df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
            for i in range(len(row.geometry)):
                df_temp.loc[i, 'geometry'] = row.geometry[i]
            outdf = outdf.append(df_temp, ignore_index=True)

        outdf.reset_index(drop=True, inplace=True)
        return outdf

    def explode(self,df):
        """

        :param df: geopandas
        :return: geopandas without multipolygon or multilinestring
        """

        if(self.check_linestring(df)):
            out_df = self.explode_all_ms(df)
            return out_df
        else:
            out_df = self.explode_mp(df)
            return out_df

    def check_linestring(self,df):
        """
        Check whether the geopandas contain linestring or not
        :param df: geopandas
        :return: boolean true or false
        """
        type_ = np.unique(df.geom_type)
        if ("Polygon" in type_):
            return False
        elif ("LineString" in type_):
            return True

    def prepare_labels(self, df, class_dictionary, id_col, label_name,drop_other_classes: bool=True):

        """
        Prepare labels for group of dataset
        :param df: geo dataframe
        :param class_dictionary : Dictionary keys are classes and values are list of ids
        :param id_col : string, column name which has id
        :param label_name : string, column name contains resulted labels
        :param drop_other_classes : boolean value whether to delete other classes in dataset or not
        :return: Classified Dataframe
        """

        if drop_other_classes is True:
            classes_to_drop = [v for values in class_dictionary.values() for v in values]
            df = df[df[id_col].isin(classes_to_drop)].copy()

        rcl_dict = {}
        rcl_dict_id = {}
        for i, (key, value) in enumerate(class_dictionary.items(), 1):
            for v in value:
                rcl_dict[v] = key
                rcl_dict_id[v] = i

        df[f'r_{label_name}'] = df[id_col].copy().map(rcl_dict)  # map name first, id second!
        df[f'r_{id_col}'] = df[id_col].map(rcl_dict_id)

        return df

    def keep_biggest_poly(self, df):
        """Replaces MultiPolygons or Mutlilines with the biggest polygon or lines contained in the MultiPolygon or Multilines."""

        if(self.check_linestring(df)):
            row_idxs_mp = df.index[df.geometry.geom_type == 'MultiLineString'].tolist()
            for idx in row_idxs_mp:
                mp = df.loc[idx].geometry
                line_lengths = [p.length for p in mp]
                max_lengths_poly = mp[line_lengths.index(max(line_lengths))]
                df.loc[idx, 'geometry'] = max_lengths_poly
            return df
        else:
            row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()
            for idx in row_idxs_mp:
                mp = df.loc[idx].geometry
                poly_areas = [p.area for p in mp]
                max_area_poly = mp[poly_areas.index(max(poly_areas))]
                df.loc[idx, 'geometry'] = max_area_poly
            return df

    def clip(self,df,clip_poly,keep_biggest_poly_: bool = False):

        """Filter and clip geodataframe to clipping geometry.

        The clipping geometry needs to be in the same projection as the geodataframe.

        Args:
            df: input geodataframe

        Returns:
            Result geodataframe.
        """

        df = df[df.geometry.intersects(clip_poly)].copy()
        # it creates Multiline or Multipolygon In order to rectify the Multiline or multipolygon, bigger one should
        # be chosen or new datapoint should be embedded
        df.geometry = df.geometry.apply(lambda _p: _p.intersection(clip_poly))
        # df = gpd.overlay(df, clip_poly, how='intersection')  # Slower.

        if(keep_biggest_poly_):
            return self.keep_biggest_poly(df)
        else:
            return df

    def save_shapes(self,df,output_file):
        """
        Save processed geopandas in processed directory
        :param df: geopandas
        """
        import os
        df.to_file(os.path.join(st.processed_data_directory,output_file), driver='ESRI Shapefile')

















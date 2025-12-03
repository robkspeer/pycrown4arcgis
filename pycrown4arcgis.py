"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
-----------------------------------------------------------------------
Copyright: 2018, Jan Zörner
Licence: GNU GPLv3
"""

import time
import warnings
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import random

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import cdist

from skimage.segmentation  import watershed
from skimage.filters import threshold_otsu

from osgeo import gdal
from osgeo import osr

from shapely.geometry import Point, Polygon

from rasterio.features import shapes as rioshapes

import laspy

import os
os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'

import arcpy
arcpy.env.overwriteOutput = True

from pycrown4arcgis import _crown_dalponte_numba
from pycrown4arcgis import _crown_dalponteCIRC_numba

gdal.UseExceptions()
warnings.filterwarnings('ignore')


class NoTreesException(Exception):
    """ Raised when no tree detected """
    pass


class GDALFileNotFoundException(Exception):
    """ Raised when GDAL file not found """
    pass


class PyCrown:

    __author__ = "Dr. Jan Zörner"
    __copyright__ = "Copyright 2018, Jan Zörner"
    __credits__ = ["Jan Zörner", "John Dymond", "James Shepherd", "Ben Jolly"]
    __license__ = "GNU GPLv3"
    __version__ = "0.1"
    __maintainer__ = "Jan Zörner"
    __email__ = "zoernerj@landcareresearch.co.nz"
    __status__ = "Development"

    def __init__(self, chm_file, dtm_file, dsm_file, las_file=None,
                 outpath=None, suffix=None):
        """ PyCrown class

        Parameters
        ----------
        chm_file :  str
                    Path to Canopy Height Model
        dtm_file :  str
                    Path to Digital Terrain Model
        dsm_file :  str
                    Path to Digital Surface Model
        las_file :  str
                    Path to LAS (LiDAR point cloud) file
        outpath  :  str, optional
                    Output directory
        suffix   :  str, optional
                    text appended to output file names
        """
        print("\nCREATING PYCROWN OBJECT:")
        suffix = f'_{suffix}' if suffix else ''

        self.outpath = Path(outpath) if outpath else Path('./')
        self.chm = None
        self.crowns = None
        self.tree_markers = None
        self.tt_corrected = None

        # Load the CHM
        print("--> Loading input files...")
        self.chm_file = Path(chm_file)
        try:
            chm_gdal = gdal.Open(str(self.chm_file), gdal.GA_ReadOnly)
        except RuntimeError as e:
            raise IOError(e)
        proj = osr.SpatialReference(wkt=chm_gdal.GetProjection())
        self.epsg = int(proj.GetAttrValue('AUTHORITY', 1))
        self.srs = arcpy.Describe(str(self.chm_file)).spatialReference
        self.geotransform = chm_gdal.GetGeoTransform()
        self.resolution = abs(self.geotransform[-1])
        self.ul_lon = chm_gdal.GetGeoTransform()[0]
        self.ul_lat = chm_gdal.GetGeoTransform()[3]
        self.chm0 = chm_gdal.GetRasterBand(1).ReadAsArray()
        chm_gdal = None

        # Load the DTM
        try:
            self.dtm_file = Path(dtm_file)
        except RuntimeError as e:
            raise IOError(e)
        dtm_gdal = gdal.Open(str(self.dtm_file), gdal.GA_ReadOnly)
        self.dtm = dtm_gdal.GetRasterBand(1).ReadAsArray()
        dtm_gdal = None

        # Load the DSM
        try:
            self.dsm_file = Path(dsm_file)
        except RuntimeError as e:
            raise IOError(e)
        dsm_gdal = gdal.Open(str(self.dsm_file), gdal.GA_ReadOnly)
        self.dsm = dsm_gdal.GetRasterBand(1).ReadAsArray()
        dsm_gdal = None

        # Load the LiDAR point cloud
        self.lidar_in_crowns = None
        self.las = None
        if las_file:
            self._load_lidar_points_cloud(las_file)
        print("<-- Input Files Loaded.")

        print("<-- Set up PyCrown trees object...")
        self.trees = pd.DataFrame(columns=[
            'top_height',
            'top_elevation',
            'top_cor_height',
            'top_cor_elevation',
            'crown_poly_raster',
            'crown_poly_smooth',
            'top_cor',
            'top',
            'tt_corrected',
            'tree_id'
        ])

        self.trees = self.trees.astype(dtype={
            'top_height': 'float',
            'top_elevation': 'float',
            'top_cor_height': 'float',
            'top_cor_elevation': 'float',
            'crown_poly_raster': 'object',
            'crown_poly_smooth': 'object',
            'top_cor': 'object',
            'top': 'object',
            'tt_corrected': 'int',
            'tree_id': 'int'
        })

        print("<-- Created.")

    def _load_lidar_points_cloud(self, fname):
        """ Loads LiDAR dataset

        Parameters
        ----------
        fname :   str
                  Path to LiDAR dataset (.las or .laz-file)
        """
        print("    - Reading .las file:")
        las = laspy.read(str(fname))

        print("        - Filtering las classification 2 (ground) and 6 (building):")
        ## Builds a list from 0 - 18
        classification_list = list(range(19))
        ## class 2 = ground, class 6 = Building
        values_to_remove = [2, 6]
        ## list comprehension to get rid of the values_to_remove
        desired_classifications = [item for item in classification_list if item not in values_to_remove]
        ## only include the desired_classifications
        mask = (las.classification == desired_classifications[0])
        for classification_value in desired_classifications[1:]:
            mask = mask | (las.classification == classification_value)
        las = las.points[mask]
        print("        - Filtered.")

        print("        - Loading las into DataFrame:")
        lidar_points = np.array((
            las.x, las.y, las.z, las.intensity, las.return_num,
            las.classification
        )).transpose()
        colnames = ['x', 'y', 'z', 'intensity', 'return_num', 'classification']
        self.las = pd.DataFrame(lidar_points, columns=colnames)
        print("        - las points loaded.")

    def _check_empty(self):
        """ Helper function raising an Exception if no trees present

        Raises
        ------
        NoTreesException
            raises Exception if no trees present
        """
        if self.trees.empty:
            raise NoTreesException

    def _to_lonlat(self, pix_x, pix_y, resolution):
        ''' Convert pixel coordinates to longitude/latitude

        Parameters
        ----------
        pix_x :      int, float, ndarray
                     Column coordinate of raster
        pix_y :      int, float, ndarray
                     Row coordinate of raster
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        tuple
            longitude(s), latitude(s)
        '''
        lon = self.ul_lon + (pix_x * resolution)
        lat = self.ul_lat - (pix_y * resolution)
        return lon, lat

    def _to_colrow(self, lon, lat, resolution):
        ''' Convert longitude/latitude to pixel coordinates
        returns either tuple of floats or 2xn ndarray

        Parameters
        ----------
        lon :        int, float, ndarray, (pandas) Series
                     Longtitude
        lat :        int, float, ndarray, (pandas) Series
                     Latitude
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        tuple
            Column/Row coordinate as floats
        or:
        ndarray
            Column/Row coordinate as 2xn ndarray
        '''
        x = (lon - self.ul_lon) / resolution
        y = (self.ul_lat - lat) / resolution
        if isinstance(x, type(y)):
            if isinstance(x, float):
                return int(x), int(y)
            if isinstance(x, (np.ndarray, pd.Series)):
                return np.array([x, y], dtype=int)
        else:
            raise Exception("Can't handle different input types for x, y.")

    def _get_z(self, lon, lat, band, resolution):
        """ Returns data from raster band for coordinate location(s)

        Parameters
        ----------
        lon :        int, float, ndarray, (pandas) Series
                     Longtitude
        lat :        int, float, ndarray, (pandas) Series
                     Latitude
        band :       ndarray
                     raster layer (e.g., CHM or DSM)
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        float
            raster value at longitude/latitude position
        """
        x, y = self._to_colrow(lon, lat, resolution)
        return band[y, x]

    def _tree_lonlat(self, loc='top'):
        ''' returns longitude/latitude of tree tops

        Parameters
        ----------
        loc :    str, optional
                 initial or corrected tree top location: `top` or `top_cor`

        Returns
        -------
        tuple
            ndarrays of longitude(s), latitude(s) of tree tops
        '''
        lons = np.array([tree[1][loc].x for tree in self.trees.iterrows()])
        lats = np.array([tree[1][loc].y for tree in self.trees.iterrows()])
        return lons, lats


    def _tree_colrow(self, loc, resolution):
        """ returns column/row of tree tops

        Parameters
        ----------
        loc :        str, optional
                     initial or corrected tree top location: `top` or `top_cor`
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        ndarray
            2xn ndarray of column(s), row(s) positions of tree tops
        """
        return self._to_colrow(np.array([tree.x for tree in self.trees[loc]]),
                               np.array([tree.y for tree in self.trees[loc]]),
                               resolution).astype(np.int32)

    def _watershed(self, inraster, th_tree=1.4):
        """ Simple implementation of a watershed tree crown delineation

        Parameters
        ----------
        inraster :   ndarray
                     raster of height values (e.g., CHM)
        th_tree :    float
                     minimum height of tree crown

        Returns
        -------
        ndarray
            raster of individual tree crowns
        """
        inraster_mask = inraster.copy()
        inraster_mask[inraster <= th_tree] = 0
        raster = inraster.copy()
        raster[np.isnan(raster)] = 0.
        labels = watershed(-raster, self.tree_markers, mask=inraster_mask)
        return labels

    def _screen_crowns(self, cond):
        """ Remove crowns outside tile from crowns raster and reindex
        the remaining ones

        Parameters
        ----------
        cond :    list
                  list of booleans. Keep trees/crowns with True
        """
        counter = 1
        for idx, valid in enumerate(cond):
            if valid:
                self.crowns[self.crowns == idx + 1] = counter
                counter += 1
            else:
                self.crowns[self.crowns == idx + 1] = 0.

    @staticmethod
    def _get_kernel(radius=5, circular=False):
        """ returns a block or disc-shaped filter kernel with given radius

        Parameters
        ----------
        radius :    int, optional
                    radius of the filter kernel
        circular :  bool, optional
                    set to True for disc-shaped filter kernel, block otherwise

        Returns
        -------
        ndarray
            filter kernel
        """
        if circular:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            return x**2 + y**2 <= radius**2
        else:
            return np.ones((int(radius), int(radius)))

    def _smooth_raster(self, raster, ws, circular=False):
        """ Smooth a raster with a median filter

        Parameters
        ----------
        raster :      ndarray
                      raster to be smoothed
        ws :          int
                      window size of smoothing filter
        circular :    bool, optional
                      set to True for disc-shaped filter kernel, block otherwise

        Returns
        -------
        ndarray
            smoothed raster
        """
        return filters.median_filter(
            raster, footprint=self._get_kernel(ws, circular=circular))

    def get_tree_height_elevation(self, loc='top'):
        ''' Sets tree height and elevation in tree dataframe

        Parameters
        ----------
        loc :    str, optional
                 initial or corrected tree top location: `top` or `top_cor`
        '''
        location = "TREE TOPS" if loc == "top" else "CORRECTED TREE TOPS" 
        print(f"\nGET TREE HEIGHT AND ELEVATION FOR {location}:")
        lons, lats = self._tree_lonlat(loc)
        print("--> Set height to CHM value, set elevation to DTM value...")
        self.trees[f'{loc}_height'] = self._get_z(
            lons, lats, self.chm, self.resolution)
        self.trees[f'{loc}_elevation'] = self._get_z(
            lons, lats, self.dtm, self.resolution)
        print(f"<-- Height and elevation set for {location.lower()}.")

    def filter_chm(self, ws, ws_in_pixels=True, circular=False):
        ''' Pre-process the canopy height model (smoothing and outlier removal).
        The original CHM (self.chm0) is not overwritten, but a new one is
        stored (self.chm).

        Parameters
        ----------
        ws :            int
                        window size of smoothing filter in metre (set in_pixel=True, otherwise)
        ws_in_pixels :  bool, optional
                        sets ws in pixel
        circular :      bool, optional
                        set to True for disc-shaped filter kernel, block otherwise
        '''
        print("\nFILTER CHM TO SMOOTH AND REMOVE OUTLIERS:")
        if not ws_in_pixels:
            if ws % self.resolution:
                raise Exception("Image filter size not an integer number. Please check if image resolution matches filter size (in metre or pixel).")
            else:
                ws = int(ws / self.resolution)

        print("--> Applying a median filter (scipy.ndimage.filters.median_filter)...")
        self.chm = self._smooth_raster(self.chm0, ws, circular=circular)
        self.chm0[np.isnan(self.chm0)] = 0
        zmask = (self.chm < 0.5) | np.isnan(self.chm) | (self.chm > 60.)
        self.chm[zmask] = 0
        print("<-- CHM Filtered.")

    def tree_detection(self, raster, resolution=None, ws=20, hmin=1.4,
                       return_trees=False, ws_in_pixels=True):
        ''' Detect individual trees from CHM raster based on a maximum filter.
        Identified trees are either stores as list in the tree dataframe or
        returned as ndarray.

        Parameters
        ----------
        raster :        ndarray
                        raster of height values (e.g., CHM)
        resolution :    int, optional
                        resolution of raster in m
        ws :            float
                        moving window size (in metre) to detect the local maxima
        hmin :          float
                        Minimum height of a tree. Threshold below which a pixel
                        or a point cannot be a local maxima
        return_trees :  bool
                        set to True if detected trees shopuld be returned as
                        ndarray instead of being stored in tree dataframe
        ws_in_pixels :  bool
                        sets ws in pixel

        Returns
        -------
        ndarray (optional)
            nx2 array of tree top pixel coordinates
        '''
        print("\nDETECTING INDIVIDUAL TREES.")
        if not isinstance(raster, np.ndarray):
            raise Exception("Please provide an input raster as numpy ndarray.")

        resolution = resolution if resolution else self.resolution

        if not ws_in_pixels:
            if ws % resolution:
                raise Exception("Image filter size not an integer number. Please check if image resolution matches filter size (in metre or pixel).")
            else:
                ws = int(ws / resolution)

        tt = time.time()
        timeit = "<-- Tree detection took {:.2f}s:"

        print("--> Apply scipy.ndimage.filters.maximum_filter to find maximum value within window size...")
        raster_maximum = filters.maximum_filter(
            raster,
            footprint = self._get_kernel(ws, circular=True)
        )
        tree_maxima = raster == raster_maximum

        # remove tree tops lower than minimum height
        tree_maxima[raster <= hmin] = 0

        # label each tree
        self.tree_markers, num_objects = ndimage.label(tree_maxima)

        print("<-- Filter applied.")

        if num_objects == 0:
            raise NoTreesException

        print("--> When multiple pixels in the window have the same maxima,\n    apply scipy.ndimage.center_of_mass to find the weighted average...")
        # if canopy height is the same for multiple pixels,
        # place the tree top in the center of mass of the pixel bounds
        yx = np.array(
                ndimage.center_of_mass(
                    raster, self.tree_markers, range(1, num_objects+1)
                ), dtype=np.float32
            ) + 0.5
        xy = np.array((yx[:, 1], yx[:, 0])).T

        trees = [Point(*self._to_lonlat(xy[tidx, 0], xy[tidx, 1], resolution))
                 for tidx in range(len(xy))]

        if return_trees:
            return np.array(trees, dtype=object), xy
        else:
            df = pd.DataFrame(np.array([trees, trees], dtype='object').T,
                              dtype='object', columns=['top_cor', 'top'])
            ## hard code index to tree_id column, because in the end
            ## some trees don't have crowns and some crowns don't have trees.
            ## this will keep the tree point, crowns, and LAS all with the same ID no matter what gets filtered.
            df["tree_id"] = df.index.values
            self.trees = pd.concat([self.trees, df])
        print(f"<-- Applied; Tree Detection Complete:\n    - Number of trees detected: {len(self.trees)}")
        print(timeit.format(time.time() - tt))
        self._check_empty()

    def crown_delineation(self, algorithm, loc='top', th_tree=1.4, **kwargs):
        """ Function calling external crown delineation algorithms

        Parameters
        ----------
        algorithm :  str
                     crown delineation algorithm to be used, choose from:
                      'dalponte_numba',
                      'dalponteCIRC_numba',
                      'watershed_skimage'
        loc :        str, optional
                     tree seed position: `top` or `top_cor`
        th_tree :    float
                     minimum height of tree seed (in m)
        th_seed :    float
                     factor 1 for minimum height of tree crown
        th_crown :   float
                     factor 2 for minimum height of tree crown
        max_crown :  float
                     maximum radius of tree crown (in m)

        Returns
        -------
        ndarray
            raster of individual tree crowns
        """
        print("\nDELINEATE CROWNS:")

        # get the tree seeds (starting points for crown delineation)
        seeds = self._tree_colrow(loc, self.resolution)
        inraster = kwargs.get('inraster')

        if not isinstance(inraster, np.ndarray):
            inraster = self.chm
        else:
            inraster = inraster

        if kwargs.get('max_crown'):
            max_crown = kwargs['max_crown'] / self.resolution

        print(f"--> Delineating crowns using the {algorithm} algorithm...")
        print(f"    - This may take a minute depending on CHM resolution...")
        timeit = "<-- Delineation took {:.2f}s:"
        if algorithm == 'dalponte_cython':
            print("dalponte_cython algorithm not available. using dalponteCIRC_numba instead.")
            algorithm = 'dalponteCIRC_numba'

        elif algorithm == 'dalponte_numba':
            tt = time.time()
            crowns = _crown_dalponte_numba.crown_dalponte(
                inraster, seeds,
                th_seed=float(kwargs['th_seed']),
                th_crown=float(kwargs['th_crown']),
                th_tree=float(th_tree),
                max_crown=float(max_crown)
            )
            print(timeit.format(time.time() - tt))

        elif algorithm == 'dalponteCIRC_numba':
            tt = time.time()
            crowns = _crown_dalponteCIRC_numba.crown_dalponteCIRC(
                inraster, seeds,
                th_seed=float(kwargs['th_seed']),
                th_crown=float(kwargs['th_crown']),
                th_tree=float(th_tree),
                max_crown=float(max_crown)
            )
            print(timeit.format(time.time() - tt))

        elif algorithm == 'watershed_skimage':
            tt = time.time()
            crowns = self._watershed(
                inraster, th_tree=float(th_tree)
            )
            print(timeit.format(time.time() - tt))

        self.crowns = np.array(crowns, dtype=np.int32)

    def correct_tree_tops(self, check_all=False):
        """ Correct the location of tree tops in steep terrain.
        Tree dataframe is updated with corrected tree top positions (`top_cor`).

        Parameters
        ----------
        check_all :    bool, optional
                       set to True if all trees should be corrected, ignoring
                       whether they are located on steep terrain
        """

        print(f"\nCORRECT TREE TOPS FOR STEEP TERRAIN:")

        # calculate center of mass of crowns
        comass = np.array(
            ndimage.center_of_mass(self.crowns, self.crowns,
                                   range(1, self.crowns.max() + 1))
        )

        corr_n = 0
        corr_dsm = 0
        corr_com = 0

        print("--> Check if tree top is too far down-slope compared to crown_mean...")
        print("    - This may take a while depending on how many trees were delineated...")
        start_time = time.time()
        ## TODO: provide updates on progress.
        for tidx in range(len(self.trees)):
            tree = self.trees.iloc[tidx]
            col, row = self._to_colrow(tree['top'].x, tree['top'].y,
                                       self.resolution)
            rcindices = np.where(self.crowns == tidx + 1)
            dtm_mean = np.nanmean(self.dtm[rcindices])
            dtm_std = np.nanstd(self.dtm[rcindices])
            dsm_max = np.nanmax(self.dsm[rcindices])

            if np.isnan(dtm_mean) or np.isnan(dsm_max):
                self.trees.tt_corrected.iloc[tidx] = -1
                continue

            if self.dtm[row, col] <= (dtm_mean - dtm_std) or check_all:

                # find highest DSM location in crown
                midx = np.where(self.dsm[rcindices] == dsm_max)[0][0]
                dsmhigh = np.array((rcindices[0][midx] + 0.5,
                                    rcindices[1][midx] + 0.5))

                # calculate map distances
                distances = cdist(np.stack(rcindices, axis=1),
                                  comass[tidx][np.newaxis])
                dist_dh_com = cdist(dsmhigh[np.newaxis],
                                    comass[tidx][np.newaxis])

                # assign high point position from DSM if new location is not
                # too far from centre of mass of the crown (1), in the latter case
                # place the tree top at the centre of mass (2)
                corr_n += 1

                if dist_dh_com <= (1. * np.nanmean(distances)):
                    cor_col, cor_row = dsmhigh[1], dsmhigh[0]
                    corr_dsm += 1
                    self.trees.tt_corrected.iloc[tidx] = 1 # Corrected to DSM location.
                else:
                    cor_col, cor_row = comass[tidx][1], comass[tidx][0]
                    corr_com += 1
                    self.trees.tt_corrected.iloc[tidx] = 2 # Corrected to centre of mass

                # Set new tree top height
                self.trees.top_cor.iloc[tidx] = \
                    Point(*self._to_lonlat(cor_col, cor_row, self.resolution))

            else:
                self.trees.tt_corrected.iloc[tidx] = 0 # No correction.
        print(f"<-- Corrections took {time.time() - start_time}s:")

        if len(self.trees) > 0:
            print(f'    - Tree tops corrected: {round(100 * corr_n / len(self.trees), 2)}%')
            print(f'    - Corrected tree top by moving to DSM high point: {corr_dsm}')
            print(f'    - Corrected tree top by moving Center of Mass: {corr_com}')
        return corr_dsm, corr_com

    def crowns_to_polys_raster(self):
        ''' Converts tree crown raster to individual polygons and stores them
        in the tree dataframe
        '''
        print("\nCONVERTING TREE CROWN RASTERS TO POLYGONS:")
        polys = []
        print("--> Convert pixel coordinates to lat/long,\n    create a path from points for the polygon edges,\n    and create a shapely.geometry.Polygon from the edges...")
        for feature in rioshapes(self.crowns, mask=self.crowns.astype(bool)):
            # Convert pixel coordinates to lon/lat
            edges = feature[0]['coordinates'][0].copy()
            for i in range(len(edges)):
                edges[i] = self._to_lonlat(*edges[i], self.resolution)

            polys.append(Polygon(edges))

        ## TODO: align polygons to their actual trees.
        ## Issue: polygons not necessarily digitized into their array in the same order as trees.
        #print("\n\npolys:")
        #print(type(polys)) ## list
        #print(polys)
        ## ie [<POLYGON ((233139.291 461336.162, 233139.291 461336.062, 233139.591 461336.0...>, <POLYGON ((233136.591 461336.062,
        #for poly in polys:
        #    print(poly)
        """
        Fix by:
        creating arcpy polygons from [polys]
        creating arcpy points from self.trees.
        
        identity or spatial join points and polygons, to get tree_id associated with poly
        convert arcpy polygons, in order of tree_id, back to Polygon(edges)
        only then set self.trees.crown_poly_raster to polys.

        """

        ## output crown polys
        crown_polys_temp = arcpy.management.CreateFeatureclass(
            out_path = "memory",
            out_name = "crown_polys_temp",
            geometry_type = "POLYGON",
            has_z = "ENABLED",
            spatial_reference = self.srs
        )
        fields = ["SHAPE@"]
        with arcpy.da.InsertCursor(crown_polys_temp, fields) as cursor:
            for poly in polys:
                ## Shapely polygon to geojson:
                polygon_geojson = poly.__geo_interface__
                ## Move geojson inot an array of points.
                if polygon_geojson["type"] == "Polygon":
                    polygon_points = arcpy.Array()
                    coords = polygon_geojson["coordinates"][0]
                    for coord in coords:
                        polygon_points.add(arcpy.Point(coord[0], coord[1]))
                    new_polygon = arcpy.Polygon(polygon_points, self.srs)
                    cursor.insertRow([new_polygon])

        ## output tree points
        tree_points_temp = arcpy.management.CreateFeatureclass(
            out_path = "memory",
            out_name = "tree_points_temp",
            geometry_type = "POINT",
            has_z = "ENABLED",
            spatial_reference = self.srs
        )
        arcpy.management.AddField(tree_points_temp, "tree_id", "Long")
        fields = ["SHAPE@XYZ", "tree_id"]
        with arcpy.da.InsertCursor(tree_points_temp, fields) as cursor:
            for tree in range(len(self.trees)):
                tree = self.trees.iloc[tree]
                x_coord = tree["top"].x
                y_coord = tree["top"].y
                z_coord = tree["top_elevation"]
                tree_id = int(tree["tree_id"])
                point = arcpy.Point(float(x_coord), float(y_coord), float(z_coord))
                point_geometry = arcpy.PointGeometry(point, self.srs)

                # Insert the row
                cursor.insertRow((point_geometry, tree_id))

        crown_point_sj = arcpy.analysis.SpatialJoin(
            target_features=crown_polys_temp,
            join_features=tree_points_temp,
            out_feature_class=r"memory\crown_point_sj",
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_ALL",
            match_option="CONTAINS",
        )

        crown_point_sj_sorted = arcpy.management.Sort(
            in_dataset = crown_point_sj,
            out_dataset = r"memory\crown_point_sj_sorted",
            sort_field = "tree_id"
        )

        polys.clear()
        with arcpy.da.SearchCursor(crown_point_sj_sorted, ["SHAPE@", "tree_id"]) as searchcursor:
            for crown in searchcursor:
                arcpy_array = arcpy.Array(crown[0])
                arcpy_polygon = arcpy.Polygon(arcpy_array)
                polygon_points = []
                for part in arcpy_polygon:
                    for point in part:
                        if point:
                            polygon_points.append((point.X, point.Y))
                shapely_polygon = Polygon(polygon_points)
                polys.append(shapely_polygon)

        self.trees.crown_poly_raster = polys
        print("<-- Tree crowns converted from raster to polys.")

    def crowns_to_polys_smooth(self,
                               store_las=True,
                               output_las_name="trees.las",
                               thin_perc=None,
                               first_return=False):
        """ Smooth crown polygons using Dalponte & Coomes (2016) approach:
        Builds a convex hull around first return points (which lie within the
        rasterized crowns).
        Optionally, the trees in the LiDAR point cloud are classified based on
        the generated convex hull.

        Parameters
        ----------
        store_las :         bool
                            set to True if LiDAR point clouds should be classified
                            and stored externally
        output_las_name:    str
                            name of las to be saved
        thin_perc :         None or int
                            percentage amount by how much the point cloud should be
                            thinned out randomly
        first_return :      bool
                            use first return points to create convex hull (all
                            points otherwise)
        """
        print("\nSMOOTHING RASTER POLYGONS USING A CONVEX HULL:")
        if thin_perc:
            thin_size = floor(len(self.las) * (1 - thin_perc))
            lidar_geodf = self.las.sample(n=thin_size)
        else:
            lidar_geodf = self.las

        print("--> Converting raster crowns to shapely polygons...")
        ## use the corrected self.trees.crown_poly_raster geometry,
        ## not the self.crowns polygons which don't align with tree_id
        crown_geodf = gpd.GeoDataFrame(
            pd.DataFrame(self.trees["tree_id"].to_numpy(), columns=["tree_id"]),
            crs=f'epsg:{self.epsg}', geometry=self.trees["crown_poly_raster"]
        )
        print("<-- Converted.")

        print("--> Converting LAS point cloud to shapely points...")
        geometry = [Point(xy) for xy in zip(lidar_geodf.x, lidar_geodf.y)]
        lidar_geodf = gpd.GeoDataFrame(lidar_geodf, crs=f'epsg:{self.epsg}',
                                       geometry=geometry)
        print("<-- Converted.")

        print("--> Use geopandas.sjoin to spatial join LiDAR points to corresponding crowns...")
        lidar_in_crowns = gpd.sjoin(lidar_geodf, crown_geodf,
                                    predicate='within', how="inner")
        lidar_tree_class = np.zeros(lidar_in_crowns['index_right'].size)
        lidar_tree_mask = np.zeros(lidar_in_crowns['index_right'].size,
                                   dtype=bool)
        print("<-- Attached.")

        print("--> Creating convex hull around first return points...")
        polys = []
        for tree_index in range(len(self.trees)):
            bool_indices = lidar_in_crowns['tree_id'] == self.trees.iloc[tree_index].tree_id
            lidar_tree_class[bool_indices] = self.trees.iloc[tree_index].tree_id
            points = lidar_in_crowns[bool_indices]
            # check that not all values are the same
            if len(points.z) > 1 and not np.allclose(points.z,
                                                     points.iloc[0].z):
                points_threshold = points.z.to_numpy()
                points = points[points.z >= threshold_otsu(points_threshold)]
                if first_return:
                    points = points[points.return_num == 1]  # first returns
            crown_unary_union = points.unary_union
            lidar_tree_mask[bool_indices] = lidar_in_crowns[bool_indices].within(crown_unary_union)
            hull = points.unary_union.convex_hull
            polys.append(hull)
        self.trees.crown_poly_smooth = polys
        print("<-- Created.")

        if store_las:
            print("\nEXPORTING LAS BASED ON GENERATED CONVEX HULL:")

            os.makedirs(self.outpath, exist_ok=True)

            ## delete all files with the basename output_raster_name if it exists. if it doesn't, fail silently.
            try:
                file_basename = output_las_name.split(".", 1)[0]
                os.remove(os.path.join(str(self.outpath), file_basename))
            except OSError:
                pass

            if ".las" not in output_las_name:
                output_las_name = output_las_name + ".las"
            path_to_output_las = os.path.join(str(self.outpath), output_las_name)

            print("--> Classifying las...")
            lidar_in_crowns = lidar_in_crowns[lidar_tree_mask]
            lidar_tree_class = lidar_tree_class[lidar_tree_mask]
    
            #print("\n\n\nlidar_tree_class:")
            #print(type(lidar_tree_class))
            #print(lidar_tree_class)

            header = laspy.LasHeader()
            self.outpath.mkdir(parents=True, exist_ok=True)
            outfile = laspy.LasData(header)
            xmin = np.floor(np.min(lidar_in_crowns.x))
            ymin = np.floor(np.min(lidar_in_crowns.y))
            zmin = np.floor(np.min(lidar_in_crowns.z))
            outfile.header.offset = [xmin, ymin, zmin]
            outfile.header.scale = [0.001, 0.001, 0.001]
            outfile.x = lidar_in_crowns.x
            outfile.y = lidar_in_crowns.y
            outfile.z = lidar_in_crowns.z
            outfile.user_data = lidar_tree_class
            outfile.intensity = lidar_tree_class
            print("<-- Classified.")

            print("--> Saving .las file to disk...")
            outfile.write(path_to_output_las)
            print("<-- Saved.")

            print("--> Projecting .las file...")
            arcpy.management.DefineProjection(
                in_dataset = path_to_output_las,
                coor_system = self.srs
            )
            print("<-- Projected.")

        self.lidar_in_crowns = lidar_in_crowns

    def quality_control(self, all_good=False):
        """ Remove trees from tree dataframe with missing DTM/DSM data &
        crowns that are not polygons

        Parameters
        ----------
        all_good :    bool
                      set to True if all trees should pass the quality check
        """
        print("\nCONDUCTING QUALITY CONTROL:")
        if all_good:
            self.trees.tt_corrected = np.zeros(len(self.trees), dtype=int)
        else:
            cond = (
                (self.trees.tt_corrected >= 0) &
                self.trees.crown_poly_raster.apply(
                    lambda x: isinstance(x, Polygon))
            )
            self.trees = self.trees[cond]

        self._check_empty()
        print("<-- PyCrown.trees is not empty; good to continue.")

    def export_tree_tops(self, input_mask_file=None, input_mask_buffer=None, loc='top', output_shape_name="top"):
        """ Convert tree top raster indices to georeferenced 3D point shapefile

        Parameters
        ----------
        loc :               str, optional
                            tree seed position: `top` or `top_cor`
        output_shape_name : str
                            name of shapefile to be saved
        """
        print(f"\nEXPORTING SHAPEFILE {output_shape_name}:")
        os.makedirs(self.outpath, exist_ok=True)
        arcpy.env.workspace = str(self.outpath)
        arcpy.env.scratchWorkspace = str(self.outpath)

        ## delete all files with the basename output_raster_name if it exists. if it doesn't, fail silently.
        try:
            file_basename = output_shape_name.split(".", 1)[0]
            os.remove(os.path.join(str(self.outpath), file_basename))
        except OSError:
            pass

        if ".shp" not in output_shape_name:
            output_shape_name = output_shape_name + ".shp"
        path_to_output_shape = os.path.join(str(self.outpath), output_shape_name)

        print("--> Create shapefile...")
        output_feature_class = arcpy.management.CreateFeatureclass(
            out_path = str(self.outpath),
            out_name = output_shape_name,
            geometry_type = "POINT",
            has_z = "ENABLED",
            spatial_reference = self.srs
        )
        print("<-- Created.")

        print("--> Add fields...")
        arcpy.management.AddFields(
            in_table = output_feature_class,
            field_description =
                [
                    ["TreeID", "LONG"],
                    ["TreeHtMtr", "FLOAT"]
                ]
        )
        print("<-- Added.")

        print(f"--> Inserting rows into {output_shape_name} at {path_to_output_shape}...")
        fields = ["SHAPE@XYZ", "TreeID", "TreeHtMtr"]
        with arcpy.da.InsertCursor(output_feature_class, fields) as cursor:
            for tree_index in range(len(self.trees)):
                tree = self.trees.iloc[tree_index]
                x_coord = tree[loc].x
                y_coord = tree[loc].y
                z_coord = tree[f'{loc}_elevation']
                TreeID = tree_index
                TreeHeightMeters = round(float(tree[f'{loc}_height']), 2)
                point = arcpy.Point(float(x_coord), float(y_coord), float(z_coord))
                point_geometry = arcpy.PointGeometry(point, self.srs)

                # Insert the row
                cursor.insertRow((point_geometry, TreeID, TreeHeightMeters))
        print(f"<-- Saved {path_to_output_shape}.")

        if input_mask_file is not None and arcpy.Exists(input_mask_file):
            if input_mask_buffer is not None and input_mask_buffer > 0:
                print(f"--> Removing trees within {input_mask_buffer} pixels of mask boundary...")
                ## delete trees on the edge of the mask
                mask_to_line = arcpy.management.FeatureToLine(
                    in_features = input_mask_file,
                    out_feature_class = r"memory\mask_to_line"
                )

                trees_feature_layer = arcpy.management.MakeFeatureLayer(
                    in_features = output_feature_class,
                    out_layer = "trees_feature_layer"
                )

                arcpy.management.SelectLayerByLocation(
                    in_layer = trees_feature_layer,
                    overlap_type="INTERSECT",
                    select_features = mask_to_line,
                    search_distance = f"{self.resolution*input_mask_buffer} Meters",
                    selection_type="NEW_SELECTION"
                )

                arcpy.management.DeleteRows(
                    in_rows = trees_feature_layer
                )
                print(f"<-- Deleted boundary trees within {input_mask_buffer} pixel buffer.")

        print(f"--> Estimate DBH using power function derived from field data for Flagstaff Ponderosa Pine")
        print(f"    See: https://www.sciencedirect.com/science/article/pii/S037811272031464X")
        print(f"    See: https://www.sciencedirect.com/science/article/pii/S0378112724000185")
        print("     - Add fields...")
        arcpy.management.AddFields(
            in_table = output_feature_class,
            field_description =
            [
                ["TreeDbhCm", "FLOAT"],
                ["TreeHtFt", "FLOAT"],
                ["TreeDbhIn", "FLOAT"],
            ]
        )
        print("     - Added.")
        print("     - Calculate fields...")
        arcpy.management.CalculateField(
            in_table=path_to_output_shape,
            field="TreeDbhCm",
            expression="calcDbh(!TreeHtMtr!)",
            expression_type="PYTHON3",
            code_block=
                ("def calcDbh(TreeHtMtr):\n"
                    "    import math\n"
                    "    return round(float(0.87*(math.pow(TreeHtMtr,1.23))), 2)")
        )
        arcpy.management.CalculateField(
            in_table=path_to_output_shape,
            field="TreeHtFt",
            expression="round((!TreeHtMtr! * 3.28084), 2)"
        )
        arcpy.management.CalculateField(
            in_table=path_to_output_shape,
            field="TreeDbhIn",
            expression="round((!TreeDbhCm! / 2.54), 2)"
        )
        print("     - Calculated.")

    def export_tree_crowns(self, crowntype="crown_poly_smooth", output_shape_name="crown_poly_smooth"):
        """ Convert tree crown raster to georeferenced polygon shapefile

        Parameters
        ----------
        crowntype :         str, optional
                            choose whether the raster of smoothed version should be
                            exported: `crown_poly_smooth` or `crown_poly_raster`
        output_shape_name : str
                            name of shapefile to be saved
        """
        print(f"\nEXPORTING SHAPEFILE {output_shape_name}:")
        os.makedirs(self.outpath, exist_ok=True)
        arcpy.env.workspace = str(self.outpath)
        arcpy.env.scratchWorkspace = str(self.outpath)

        ## delete all files with the basename output_raster_name if it exists. if it doesn't, fail silently.
        try:
            file_basename = output_shape_name.split(".", 1)[0]
            os.remove(os.path.join(str(self.outpath), file_basename))
        except OSError:
            pass

        if ".shp" not in output_shape_name:
            output_shape_name = output_shape_name + ".shp"
        path_to_output_shape = os.path.join(str(self.outpath), output_shape_name)

        print("--> Create shapefile...")
        output_feature_class = arcpy.management.CreateFeatureclass(
            out_path = str(self.outpath),
            out_name = output_shape_name,
            geometry_type = "POLYGON",
            has_z = "ENABLED",
            spatial_reference = self.srs
        )
        print("<-- Created.")

        print("--> Add fields...")
        arcpy.management.AddFields(
            in_table = output_feature_class,
            field_description =
                [
                    ["TreeID", "LONG"],
                    ["TreeHtMtr", "FLOAT"]
                ]
        )
        print("<-- Added.")

        print(f"--> Inserting rows into {output_shape_name} at {path_to_output_shape}...")
        fields = ["SHAPE@", "TreeID", "TreeHtMtr"]
        with arcpy.da.InsertCursor(output_feature_class, fields) as cursor:
            for tree_index in range(len(self.trees)):
                tree = self.trees.iloc[tree_index]
                ## Shapely polygon to geojson:
                polygon_geojson = tree[crowntype].__geo_interface__

                ## Move geojson inot an array of points.
                if polygon_geojson["type"] == "Polygon":
                    array = arcpy.Array([arcpy.Point(*coords) for coords in polygon_geojson["coordinates"][0]])
                    polygon = arcpy.Polygon(array, self._screen_crowns)

                    TreeID = tree_index
                    TreeHeight = round(float(tree.top_height), 2)
                    # CorrHeight = round(float(tree.top_cor_height), 2)

                    # Insert the row
                    cursor.insertRow((polygon, TreeID, TreeHeight))
        print(f"<-- Saved {path_to_output_shape}.")

        print("--> Join TreeDbhCm, TreeHtFt, and TreeDbhIn to crowns...")
        tree_tops_shape_name = f"{output_shape_name.split('_', 1)[0]}_PyCrownTrees.shp"
        arcpy.management.JoinField(
            in_data=path_to_output_shape,
            in_field="TreeID",
            join_table=os.path.join(str(self.outpath), tree_tops_shape_name),
            join_field="TreeID",
            fields="TreeDbhCm;TreeHtFt;TreeDbhIn"
        )
        print("<-- Joined.")
        
    def export_raster(self, raster, output_raster_name, input_mask_file, res=None):
        """ Write array to raster file with gdal

        Parameters
        ----------
        raster :                ndarray
                                raster to be exported
        output_raster_name :    str
                                file name
        input_mask_file :       str, path
                                file path to a polygon feature class that will be used as the boundary mask for the analysis
        res :                   int/float, optional
                                resolution of the raster in m, if not provided the same as
                                the input CHM
        """
        print(f"\nEXPORTING RASTER {output_raster_name}:")

        os.makedirs(self.outpath, exist_ok=True)
        arcpy.env.workspace = str(self.outpath)
        arcpy.env.scratchWorkspace = str(self.outpath)

        res = res if res else self.resolution

        ## delete all files with the basename output_raster_name if it exists. if it doesn't, fail silently.
        try:
            file_basename = output_raster_name.split(".", 1)[0]
            os.remove(os.path.join(str(self.outpath), file_basename))
        except OSError:
            pass

        if ".tif" not in output_raster_name:
            output_raster_name = output_raster_name + ".tif"
        path_to_output_raster = os.path.join(str(self.outpath), output_raster_name)

        print("--> Convert NumPyArrayToRaster...")
        input_chm = arcpy.Raster(str(self.chm_file))
        raster_extent = input_chm.extent

        output_chm = arcpy.NumPyArrayToRaster(
            in_array = raster,
            lower_left_corner = arcpy.Point(raster_extent.XMin, raster_extent.YMin),
            x_cell_size = res,
            y_cell_size = res
        )
        print("<-- Converted.")

        path_to_temp_tif = os.path.join(str(self.outpath), "temp.tif")
        print(f"--> Saving temporary tif for further processing...")
        output_chm.save(path_to_temp_tif)
        print(f"<-- Saved temp.tif.")

        arcpy.management.BuildPyramidsandStatistics(
            in_workspace = path_to_temp_tif
        )

        print("--> Define Raster Projection...")
        arcpy.management.DefineProjection(
            in_dataset = path_to_temp_tif,
            coor_system = self.srs
        )
        print("<-- Defined.")

        try:
            print("--> Calculate Null raster values to 0...")
            arcpy.sa.SetNull(path_to_temp_tif, path_to_temp_tif, "VALUE = 0")
            print("<-- Calculated.")
        except:
            print("<-- arcpy.sa.SetNull failed.")

        print(f"--> Extracting {output_raster_name} to inside mask...")
        final_chm_out_raster = arcpy.sa.ExtractByMask(
            in_raster = path_to_temp_tif,
            in_mask_data = input_mask_file,
            extraction_area = "INSIDE"
        )
        final_chm_out_raster.save(path_to_output_raster)
        print(f"<-- Saved {path_to_output_raster}.")

        if arcpy.Exists(path_to_temp_tif):
            arcpy.management.Delete(path_to_temp_tif)
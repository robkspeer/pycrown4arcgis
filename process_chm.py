import arcpy
import os
import shutil
import time
from pathlib import Path
from pycrown4arcgis import PyCrown
import json

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

class ProcessCHM:
    def __init__(self, project_basename, input_las_file, input_mask_file, chm_result_directory, output_wkid = 26949, output_resolution = 5, pro_project = None):
        """ ProcessCHM class
        """
        arcpy.env.mask = input_mask_file

        target_group_layer_name = f"{project_basename} Results CHM"
        target_group_layer : arcpy.mp.Layer
        active_map : arcpy.mp.Map
        if pro_project.activeMap is not None:
            print("\n--> Prep map by removing existing CHM group layer to remove locks...")
            active_map = pro_project.activeMap
            ## If the group layer already exists in the map,
            ## remove it and save the project to remove existing locks that prevent overwriting.
            for layer in active_map.listLayers():
                if layer.isGroupLayer and layer.name == target_group_layer_name:
                    active_map.removeLayer(layer)
                    pro_project.save()
                    time.sleep(2)
            target_group_layer = active_map.createGroupLayer(target_group_layer_name)
            print("<-- Map prepped.")

        os.makedirs(chm_result_directory, exist_ok=True)
        os.makedirs(os.path.join(chm_result_directory, "Extracted"), exist_ok=True)

        input_las_path = Path(input_las_file)

        ## creates a semi-colon delimited string of numbers from 0-18 for classifications to include, except 2 and 6.
        ## ";".join(str(num) for num in ([item for item in list(range(19)) if item not in [2,6]]
        ## 2 = ground
        ## 6 = buildings
        ## Ground is needed for the DSM to calculate the Canopy Height Model, don't filter it out.
        print("\n--> Filter LAS dataset to exclude existing buildings...")
        remove_classifications = [6]
        filtered_las_dataset_layer = arcpy.management.MakeLasDatasetLayer(
            in_las_dataset = input_las_file,
            out_layer = "filtered_las_dataset_layer",
            class_code = ";".join(str(num) for num in ([item for item in list(range(19)) if item not in remove_classifications]))
        )

        extracted_target_directory = os.path.join(chm_result_directory, "Extracted")
        extracted_las = os.path.join(extracted_target_directory, f"{input_las_path.stem}_Extracted.las")
        if arcpy.Exists(extracted_las):
            arcpy.management.Delete(extracted_las)
        projected_las = os.path.join(chm_result_directory, f"{input_las_path.stem}_Projected.las")
        if arcpy.Exists(projected_las):
            arcpy.management.Delete(projected_las)

        extracted_las = arcpy.ddd.ExtractLas(
            in_las_dataset = filtered_las_dataset_layer,
            target_folder = extracted_target_directory,
            boundary = input_mask_file,
            name_suffix = "_Extracted",
            compression="NO_COMPRESSION"
        )
        print("<-- Filtered.")

        print("\n--> Project LAS dataset to spatial reference in meters...")
        projected_las = arcpy.management.ProjectLAS(
            in_las_dataset = os.path.join(extracted_target_directory, f"{input_las_path.stem}_Extracted.las"),
            target_folder = chm_result_directory,
            coordinate_system = arcpy.SpatialReference(int(output_wkid)),
            compression = "NO_COMPRESSION",
            name_modifier = "Projected"
        )
        print("<-- Projected.")

        print("\n--> Clean up intermediate LAS data...")
        if os.path.exists(extracted_target_directory):
            shutil.rmtree(extracted_target_directory)

        path_to_las = os.path.join(chm_result_directory, f"{project_basename}_Projected.las")
        if arcpy.Exists(path_to_las):
            arcpy.management.Delete(path_to_las)

        arcpy.management.Rename(
            in_data = os.path.join(chm_result_directory, f"Projected{input_las_path.stem}_Extracted.las"),
            out_data = f"{project_basename}_Projected.las"
        )
        print("<-- Cleaned up.")

        las_dataset_to_raster = os.path.join(chm_result_directory, f"{project_basename}_Projected.las")

        print("\n--> Create DSM from LAS dataset...")
        dsm_raw = arcpy.conversion.LasDatasetToRaster(
            in_las_dataset = las_dataset_to_raster,
            out_raster = os.path.join(chm_result_directory, f"{project_basename}_DSM_raw.tif"),
            value_field="ELEVATION",
            interpolation_type="BINNING MAXIMUM LINEAR",
            data_type="FLOAT",
            sampling_type="CELLSIZE",
            sampling_value=output_resolution,
            z_factor=1
        )

        dsm_out_raster = arcpy.sa.ExtractByMask(
            in_raster = dsm_raw,
            in_mask_data = input_mask_file,
            extraction_area = "INSIDE"
        )
        path_to_dsm = os.path.join(chm_result_directory, f"{project_basename}_DSM.tif")
        dsm_out_raster.save(path_to_dsm)
        print("<-- DSM created.")

        print("\n--> Create DTM from LAS dataset using only bare ground classified points...")
        filtered_dsm_las_dataset_layer = arcpy.management.MakeLasDatasetLayer(
            in_las_dataset = las_dataset_to_raster,
            out_layer = "filtered_dsm_las_dataset_layer",
            class_code = "2"
        )

        dtm_raw = arcpy.conversion.LasDatasetToRaster(
            in_las_dataset = filtered_dsm_las_dataset_layer,
            out_raster = os.path.join(chm_result_directory, f"{project_basename}_DTM_raw.tif"),
            value_field="ELEVATION",
            interpolation_type="BINNING MINIMUM LINEAR",
            data_type="FLOAT",
            sampling_type="CELLSIZE",
            sampling_value=output_resolution,
            z_factor=1
        )

        dtm_out_raster = arcpy.sa.ExtractByMask(
            in_raster = dtm_raw,
            in_mask_data = input_mask_file,
            extraction_area = "INSIDE"
        )
        path_to_dtm = os.path.join(chm_result_directory, f"{project_basename}_DTM.tif")
        dtm_out_raster.save(path_to_dtm)
        print("<-- DTM created.")

        ## delete in-memory rasters.
        arcpy.management.Delete(dsm_raw)
        arcpy.management.Delete(dtm_raw)

        print("\n--> Create CHM by subtracting DTM from DSM...")
        chm_out_raster = arcpy.ia.RasterCalculator(
            rasters = [path_to_dsm, path_to_dtm],
            input_names = ["dsm", "dtm"],
            expression = "dsm - dtm"
        )
        path_to_chm = os.path.join(chm_result_directory, f"{project_basename}_CHM.tif")
        chm_out_raster.save(path_to_chm)
        print("<-- CHM created.")

        try:
            if active_map is not None:
                print("\n--> Add new layers to map...")
                las_layer = active_map.addDataFromPath(path_to_las)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = las_layer,
                                           add_position = "TOP")
                active_map.removeLayer(las_layer)

                chm_layer = active_map.addDataFromPath(path_to_chm)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = chm_layer,
                                           add_position = "BOTTOM")
                active_map.removeLayer(chm_layer)

                dsm_layer = active_map.addDataFromPath(path_to_dsm)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = dsm_layer,
                                           add_position = "BOTTOM")
                active_map.removeLayer(dsm_layer)

                dtm_layer = active_map.addDataFromPath(path_to_dtm)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = dtm_layer,
                                           add_position = "BOTTOM")
                active_map.removeLayer(dtm_layer)
                target_group_layer.isExpanded = True
                print("<-- Layers added.")
        except Exception as e:
            print(e)
            pass

        ## delete in-memory variables to release any locks.
        del path_to_las
        del path_to_chm
        del path_to_dsm
        del path_to_dtm

        print("\n- ProcessCHM Done.")

class ProcessCrowns:
    def __init__(self,
                project_basename,
                pycrown_result_directory,
                pycrown_CHM,
                pycrown_DTM,
                pycrown_DSM,
                pycrown_LAS,
                input_mask_file,
                chm_smooth_window_size = 1,
                chm_smooth_circular = True,
                tree_detection_window_size = 1.5,
                tree_detection_min_height = 1.4,
                crown_delineation_algorithm = "watershed_skimage",
                crown_delineation_th_seed = 0.45,
                crown_delineation_th_crown = 0.55,
                crown_delineation_th_tree = 2,
                crown_delineation_max_crown = 10,
                pro_project = None):
        """ ProcessCrowns class
        """

        target_group_layer_name = f"{project_basename} Results PyCrown"
        target_group_layer : arcpy.mp.Layer
        active_map : arcpy.mp.Map
        if pro_project.activeMap is not None:
            print("\n--> Prep map by removing existing PyCrown group layer to remove locks...")
            active_map = pro_project.activeMap
            ## If the group layer already exists in the map,
            ## remove it and save the project to remove existing locks that prevent overwriting.
            for layer in active_map.listLayers():
                if layer.isGroupLayer and layer.name == target_group_layer_name:
                    active_map.removeLayer(layer)
                    pro_project.save()
                    time.sleep(2)
            target_group_layer = active_map.createGroupLayer(target_group_layer_name)
            print("<-- Map prepped.")

        os.makedirs(pycrown_result_directory, exist_ok=True)

        PC = PyCrown(
            chm_file = pycrown_CHM,
            dtm_file = pycrown_DTM,
            dsm_file = pycrown_DSM,
            las_file = pycrown_LAS,
            outpath = pycrown_result_directory
        )

        PC.filter_chm(
            ws = chm_smooth_window_size,
            ws_in_pixels = True,
            circular = chm_smooth_circular
        )

        PC.tree_detection(
            raster = PC.chm,
            ws = tree_detection_window_size,
            hmin = tree_detection_min_height
            )

        PC.crown_delineation(
            algorithm = crown_delineation_algorithm,
            loc = "top",
            th_seed = crown_delineation_th_seed,
            th_crown = crown_delineation_th_crown,
            th_tree = crown_delineation_th_tree,
            max_crown = crown_delineation_max_crown)

        PC.correct_tree_tops(check_all=True)

        PC.get_tree_height_elevation(loc='top')

        PC.get_tree_height_elevation(loc='top_cor')

        PC.crowns_to_polys_raster()

        PC.crowns_to_polys_smooth(
            store_las=True,
            output_las_name = f"{project_basename}_PyCrownTrees"
        )

        PC.quality_control()

        PC.export_raster(
            raster = PC.chm,
            output_raster_name = f"{project_basename}_PyCrownCHM",
            input_mask_file = input_mask_file
        )
        PC.export_tree_locations(
            loc = "top",
            output_shape_name = f"{project_basename}_PyCrownTrees"
        )
        PC.export_tree_locations(
            loc = "top_cor",
            output_shape_name = f"{project_basename}_PyCrownTreesCorrected"
        )
        PC.export_tree_crowns(
            crowntype = "crown_poly_raster",
            output_shape_name = f"{project_basename}_PyCrownPolyRaster"
        )
        PC.export_tree_crowns(
            crowntype = "crown_poly_smooth",
            output_shape_name = f"{project_basename}_PyCrownPolySmooth"
        )

        try:
            if active_map is not None:
                print("\n--> Add new layers to map...")
                ## TREETOPS LAYER:
                treetops_layer = active_map.addDataFromPath(os.path.join(pycrown_result_directory, f"{project_basename}_PyCrownTreesCorrected.shp"))
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = treetops_layer,
                                           add_position = "TOP")
                active_map.removeLayer(treetops_layer)

                ## CROWNS LAYER:
                polysmooth_layer = active_map.addDataFromPath(os.path.join(pycrown_result_directory, f"{project_basename}_PyCrownPolySmooth.shp"))
                ## access cartographic information model (cim) to set symbology:
                polysmooth_layer_cim = polysmooth_layer.getDefinition("V3")
                polysmooth_layer_cim_symbol_outline = polysmooth_layer_cim.renderer.symbol.symbol.symbolLayers[0]
                polysmooth_layer_cim_symbol_outline.width = 1.5
                polysmooth_layer_cim_symbol_outline.color = {
                    "type" : "CIMRGBColor",
                    "values" : [0, 0, 0, 100]
                }
                polysmooth_layer_cim_symbol_fill = polysmooth_layer_cim.renderer.symbol.symbol.symbolLayers[1]
                polysmooth_layer_cim_symbol_fill.color = {
                    "type" : "CIMRGBColor",
                    "values" : [255, 255, 255, 0]
                }
                polysmooth_layer.setDefinition(polysmooth_layer_cim)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = polysmooth_layer,
                                           add_position = "BOTTOM")
                active_map.removeLayer(polysmooth_layer)

                ## LAS LAYER:
                module_path = os.path.abspath(__file__)
                module_dir = os.path.dirname(module_path)
                layer_file_path = os.path.join(module_dir, "PyCrownLasTrees.json")

                las_layer_cim : str
                try:
                    with open(layer_file_path, 'r') as template_layer_file:
                        las_layer_cim = json.load(template_layer_file)
                except FileNotFoundError:
                    print("Error: 'PyCrownLasTrees.json' not found.")
                    raise Exception
                except json.JSONDecodeError:
                    print("Error: Invalid JSON format in 'PyCrownLasTrees.json'.")
                    raise Exception

                las_layer_cim["layerDefinitions"][0]["name"] = f"{project_basename}_PyCrownTrees.las"
                las_layer_cim["layerDefinitions"][0]["dataConnection"]["workspaceConnectionString"] = pycrown_result_directory
                las_layer_cim["layerDefinitions"][0]["dataConnection"]["dataset"] = f"{project_basename}_PyCrownTrees.las"

                path_to_new_las_layer_file = os.path.join(pycrown_result_directory, f"{project_basename}_PyCrownTrees.lyrx")
                with open(path_to_new_las_layer_file, 'w') as new_las_layer_file:
                    json.dump(las_layer_cim, new_las_layer_file, indent=2)

                treelas_layer = active_map.addDataFromPath(path_to_new_las_layer_file)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = treelas_layer,
                                           add_position = "BOTTOM")
                active_map.removeLayer(treelas_layer)

                ## CHM LAYER:
                chm_layer = active_map.addDataFromPath(os.path.join(pycrown_result_directory, f"{project_basename}_PyCrownCHM.tif"))
                # Check if the layer supports a raster colorizer
                if hasattr(chm_layer, "symbology"):
                    # Access the symbology object
                    raster_symbology = chm_layer.symbology
                    raster_symbology.colorizer.colorRamp = pro_project.listColorRamps("Elevation #6")[0]
                    chm_layer.symbology = raster_symbology
                ## access cartographic information model (cim) to set symbology:
                chm_layer_cim = chm_layer.getDefinition("V3")
                chm_layer_cim.colorizer.displayBackgroundValue = True
                chm_layer.setDefinition(chm_layer_cim)
                active_map.addLayerToGroup(target_group_layer = target_group_layer,
                                           add_layer_or_layerfile = chm_layer,
                                           add_position = "BOTTOM")
                active_map.removeLayer(chm_layer)

                target_group_layer.isExpanded = True
                print("<-- Layers added.")

        except Exception as e:
            print(f"\n\nEXCEPTION: {e}")
            raise Exception
        
        print("\n- ProcessCrowns Done.")
from utils import data, plot
def segmentation_all(image_pi, image_biosensor,erosionfactor, params):
    """
    Performs segmentation on two input images: a PI-stained image and a biosensor image.
    
    Parameters:
    image_pi (array): An image array, typically PI-stained, used for identifying cellular structures.
    image_biosensor (array): An image array from a biosensor, used for identifying specific cellular components.
    
    The function applies a series of image processing techniques including denoising, filtering,
    thresholding, and morphological operations to segment different cellular components.
    
    Returns:
    tuple: A tuple containing three elements:
        - membranes (array): Segmented membrane regions.
        - cytcorrected (array): Segmented cytosolic regions, corrected for specific signals.
        - endosomes (array): Segmented endosomal regions.
    """
    
    median_radius, max_filter_size, top_hat_radius, closing_radius1, closing_radius2, dilation_radius1,dilation_radius2, erosion_radius,vmin,vmedian,biomedian,biotophat,dontprocess = params
    # Apply segmentation with varying parameters
    denoised_image = cle.median_box(image_pi, radius_x=median_radius, radius_y=median_radius, radius_z=0)
    denoised_image2 = cle.fabs(denoised_image)
    denoised_image2 = ndimage.maximum_filter(cle.pull(denoised_image2), size=max_filter_size)
    # Top hat
    denoised_image2 = cle.top_hat_box(
        denoised_image2, radius_x=top_hat_radius, radius_y=top_hat_radius, radius_z=0)
    # Sqrt filter
    denoised_image3 = cle.sqrt(denoised_image2)
    # Otsu auto threshold
    binary1 = cle.threshold_otsu(denoised_image3)
    # Closing operation to fill gaps
    binary = cle.closing_labels(binary1, radius=closing_radius1)
    skeleton = skeletonize(cle.pull(binary))
    skeleton = (skeleton > 0).astype(np.uint8) * 255
    pruned_skeleton, segmented_img, segment_objects = prune(
        skel_img=skeleton, size=1000)
    cle_image = cle.push(pruned_skeleton)
    dilate = cle.dilate_labels(cle_image, cle_image, radius=dilation_radius1)
    dilate = cle.closing_labels(dilate, radius=closing_radius2)
    dilate = cle.erode_labels(dilate, radius=erosion_radius)
    dilate = skeletonize(cle.pull(dilate))
    cle_image = cle.push(dilate)
    dilate = cle.dilate_labels(cle_image, cle_image, radius=dilation_radius2)
    dilate = cle.closing_labels(dilate, radius=closing_radius2)
    inverted = np.asarray(dilate) == 0 * 1
    label = cle.connected_components_labeling_box(inverted)
    exclude = cle.exclude_labels_on_edges(label)
    exclude = cle.exclude_labels_outside_size_range(
        exclude, minimum_size=2500, maximum_size=100000000)

    # vacuole removal
    denoised_image = cle.median_box(
        image_biosensor, radius_x=vmedian, radius_y=vmedian, radius_z=0)
    mini = cle.minimum_box(denoised_image, radius_x=vmin, radius_y=vmin, radius_z=0)
    binary2 = cle.threshold_otsu(mini)
    inverted2 = np.asarray(binary2) == 0 * 1
    cytcorrected = cle.binary_subtract(exclude, inverted2)

    # remove out of range label
    extend = cle.extend_labels_with_maximum_radius(exclude, radius=7)
    membranes = cle.binary_subtract(extend, label)
    # Keep endosomes from cytosolic signal
    denoised_image2 = cle.median_box(
        image_biosensor, radius_x=biomedian, radius_y=biomedian)
    cyt_one = cle.divide_images(cytcorrected, cytcorrected)
    # 5 for 564, 3 for 604? and 1 for 991
    cyt_one = cle.erode_labels(cyt_one, radius=erosionfactor)
    cyt_one = cle.multiply_images(cyt_one, denoised_image2)
    if dontprocess=="true":
        denoised_image3=cyt_one
    else:
        denoised_image3 = cle.top_hat_box(
            cyt_one, radius_x=biotophat, radius_y=biotophat)
    endosomes = cle.threshold_otsu(denoised_image3)#denoised image 3
    endosomes = cle.multiply_images(cytcorrected, endosomes)
    return membranes, cytcorrected, endosomes

from skimage.io import imsave
from skimage.morphology import skeletonize
from scipy import ndimage
from plantcv.plantcv.morphology import prune
import pyclesperanto_prototype as cle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def segmentationMovie(directory,window,erosionfactor,values,params):
    lrc_directory=os.getcwd()
    median_radius, max_filter_size, top_hat_radius, closing_radius1, closing_radius2, dilation_radius1,dilation_radius2, erosion_radius,vmin,vmedian,biomedian,biotophat,dontprocess = params
    os.chdir(directory)
    # Load a time series of TIF files
    image_stack = tifffile.imread('pi.tif')
    image_stackps = tifffile.imread("fluo.tif")
    cytosol_stack = []
    endosomes_stack = []
    membrane_stack = []
    skeleton_stack = []
    i = 0
    for frame, frameps in zip(image_stack, image_stackps):
        data.update_console(
            window, "-CONSOLE-", f'generating cytosol and membrane masks frame n° {str(i)}')
        # if values["method"]=="method 1":
        # Median blur denoising
        denoised_image = cle.median_box(
            frame, radius_x=median_radius, radius_y=median_radius, radius_z=0)
        # Maximum filter
        denoised_image2 = cle.fabs(denoised_image)
        denoised_image2 = ndimage.maximum_filter(
            cle.pull(denoised_image2), size=max_filter_size)
        # Top hat
        denoised_image2 = cle.top_hat_box(
            denoised_image2, radius_x=top_hat_radius, radius_y=top_hat_radius, radius_z=0)
        # Sqrt filter
        denoised_image3 = cle.sqrt(denoised_image2)
    
        # Otsu auto threshold
        binary1 = cle.threshold_otsu(denoised_image3)
        # Closing operation to fill gaps
        binary = cle.closing_labels(binary1, radius=closing_radius1)
        skeleton = skeletonize(cle.pull(binary))
        skeleton = (skeleton > 0).astype(np.uint8) * 255
        pruned_skeleton, segmented_img, segment_objects = prune(
            skel_img=skeleton, size=1000)
        cle_image = cle.push(pruned_skeleton)
        dilate = cle.dilate_labels(cle_image, cle_image, radius=dilation_radius1)
        dilate = cle.closing_labels(dilate, radius=closing_radius2)
        dilate = cle.erode_labels(dilate, radius=erosion_radius)
        dilate = skeletonize(cle.pull(dilate))
        cle_image = cle.push(dilate)
        dilate = cle.dilate_labels(cle_image, cle_image, radius=dilation_radius2)
        dilate = cle.closing_labels(dilate, radius=closing_radius2)
        skeleton_stack.append(cle.pull(dilate))
    
        inverted = np.asarray(dilate) == 0 * 1
        label = cle.connected_components_labeling_box(inverted)
        exclude = cle.exclude_labels_on_edges(label)
        exclude = cle.exclude_labels_outside_size_range(
            exclude, minimum_size=2500, maximum_size=100000000)
    
        # Vacuole mask from sensor fluorescence
        data.update_console(window, "-CONSOLE-",
                       f'generating vacuole masks')
        denoised_image = cle.median_box(
            frameps, radius_x=vmedian, radius_y=vmedian, radius_z=0)
        mini = cle.minimum_box(
            denoised_image, radius_x=vmin, radius_y=vmin, radius_z=0)
        binary2 = cle.threshold_otsu(mini)
        inverted2 = np.asarray(binary2) == 0 * 1
        cytcorrected = cle.binary_subtract(exclude, inverted2)
    
        # remove out of range label
        cytosol_stack.append(cle.pull(cytcorrected))
        extend = cle.extend_labels_with_maximum_radius(
            exclude, radius=7)
        membranes = cle.binary_subtract(extend, label)
    
        membrane_stack.append(cle.pull(membranes))
    
        # Keep endosomes from cytosolic signal
        denoised_image2 = cle.median_box(
            frameps, radius_x=biomedian, radius_y=biomedian)
        cyt_one = cle.divide_images(cytcorrected, cytcorrected)
        # 5 for 564, 3 for 604? and 1 for 991
        cyt_one = cle.erode_labels(cyt_one, radius=erosionfactor)
        cyt_one = cle.multiply_images(cyt_one, denoised_image2)
        
        if dontprocess=="true":
            denoised_image3=cyt_one
        else:
            denoised_image3 = cle.top_hat_box(
                cyt_one, radius_x=biotophat, radius_y=biotophat)
        endosomes = cle.threshold_otsu(denoised_image3)
        endosomes = cle.multiply_images(cytcorrected, endosomes)
        endosomes_stack.append(cle.pull(endosomes))
    
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
    
        # Display images
        ax1.imshow(np.hstack((membranes, cytcorrected, endosomes)))
    
        ax1.set_axis_off()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_frame_on(False)
        plt.subplots_adjust(wspace=0, hspace=0,
                            left=0, right=1, bottom=0, top=1)
        plt.show()
    
        for item in window['-CANVAS-'].TKCanvas.pack_slaves():
            item.destroy()
        plot.draw_figure(window['-CANVAS-'].TKCanvas, fig)
        window.refresh()
    
        # Measure mean fluorescences and positions
        stat_mb = cle.statistics_of_labelled_pixels(
            frameps, membranes)
        stat_cyt = cle.statistics_of_labelled_pixels(
            frameps, endosomes)
    
        table_mb_temp = pd.DataFrame(
            stat_mb)[['label', 'mean_intensity', "centroid_x", "centroid_y"]]
        table_mb_temp = pd.concat(
            [table_mb_temp, pd.DataFrame([i]*len(table_mb_temp))], axis=1)
        table_mb_temp = table_mb_temp.rename(columns={0: "Frames"})
        table_cyt_temp = pd.DataFrame(
            stat_cyt)[['label', 'mean_intensity', "centroid_x", "centroid_y"]]
        table_cyt_temp = pd.concat(
            [table_cyt_temp, pd.DataFrame([i]*len(table_cyt_temp))], axis=1)
        table_cyt_temp = table_cyt_temp.rename(
            columns={0: "Frames"})
        if i == 0:
            table_mb = table_mb_temp  # add frame!!!!!
            table_cyt = table_cyt_temp
        else:
            table_mb = pd.concat([table_mb, table_mb_temp])
            table_cyt = pd.concat([table_cyt, table_cyt_temp])
    
        i += 1
    membrane_stack = np.stack(membrane_stack)
    cytosol_stack = np.stack(cytosol_stack)
    skeleton_stack = np.stack(skeleton_stack)
    endosomes_stack = np.stack(endosomes_stack)
    stitched_image = np.concatenate(image_stackps, axis=1)
    imsave(directory+'/membrane_stack.tif', membrane_stack)
    imsave(directory+'/cytosol_stack.tif', cytosol_stack)
    imsave(directory+'/endosomes_stack.tif', endosomes_stack)
    imsave(directory+'/skeleton_stack.tif', skeleton_stack)
    print("")
    data.jsonProof(config=f'{lrc_directory}/config.json',savepath=f'{directory}/LRC_parameters.json')
    
    if values["save_stitched"]:
        imsave(directory+'/stitchimage.tif', stitched_image)
    # ordering table
    table_mb = table_mb.sort_values(by=['label', "Frames"])
    table_cyt = table_cyt.sort_values(by=['label', "Frames"])
    os.chdir(lrc_directory)
    
    return table_mb,table_cyt,cytosol_stack

def cellChecker(cytosol_stack):
    ncells = len(np.unique(cytosol_stack[0])) - 1  # Number of cells in the first frame
    list_frames_tocorrect = []
    list_frames_new_cells = []
    list_frames_no_cells = []

    for i, frame in enumerate(cytosol_stack):
        ncells_temp = len(np.unique(frame)) - 1
        if ncells_temp < ncells:
            list_frames_tocorrect.append(i)
        elif ncells_temp > ncells:
            list_frames_tocorrect.append(i)
            list_frames_new_cells.append(i)
        elif ncells_temp == 0:
            list_frames_no_cells.append(i)

    return ncells, list_frames_tocorrect, list_frames_new_cells, list_frames_no_cells

def nCellCorrection(list_frames_tocorrect,list_frames_new_cells,list_frames_no_cells,table_cyt,table_mb,window,ncells,cytosol_stack_lenght):
    if len(list_frames_tocorrect) == 0:

        for label in np.unique(table_cyt["label"]):
            if label == 1:
                df_cyt = pd.DataFrame(
                    table_cyt[table_cyt["label"] == label])
                df_cyt = df_cyt.set_index("Frames")
                df_cyt = df_cyt.drop(
                    ["label", 'centroid_x', 'centroid_y'], axis=1)
                df_cyt = df_cyt.rename(
                    columns={"mean_intensity": "Cytosolic_signal"})

                df_mb = pd.DataFrame(
                    table_mb[table_mb["label"] == label])
                df_mb = df_mb.set_index("Frames")
                df_mb = df_mb.drop(
                    ['centroid_x', 'centroid_y'], axis=1)
                df_mb = df_mb.rename(
                    columns={"mean_intensity": "Membrane_signal"})

                df_ratio = pd.concat([df_mb, df_cyt], axis=1)
                ratio_temp = df_ratio["Membrane_signal"].div(
                    df_ratio["Cytosolic_signal"], fill_value=-1)
                df_ratio = pd.concat(
                    [df_ratio, ratio_temp], axis=1)
                df_ratio = df_ratio.rename(
                    columns={0: "ratio_mb/cyt"})
                df_ratio = df_ratio[[
                    "label", "Membrane_signal", "Cytosolic_signal", "ratio_mb/cyt"]]

            else:
                df_cyt = pd.DataFrame(
                    table_cyt[table_cyt["label"] == label])
                df_cyt = df_cyt.set_index("Frames")
                df_cyt = df_cyt.drop(
                    ["label", 'centroid_x', 'centroid_y'], axis=1)
                df_cyt = df_cyt.rename(
                    columns={"mean_intensity": "Cytosolic_signal"})

                df_mb = pd.DataFrame(
                    table_mb[table_mb["label"] == label])
                df_mb = df_mb.set_index("Frames")
                df_mb = df_mb.drop(
                    ['centroid_x', 'centroid_y'], axis=1)
                df_mb = df_mb.rename(
                    columns={"mean_intensity": "Membrane_signal"})

                df_ratio2 = pd.concat([df_mb, df_cyt], axis=1)
                ratio_temp = df_ratio2["Membrane_signal"].div(
                    df_ratio2["Cytosolic_signal"], fill_value=-1)
                df_ratio2 = pd.concat(
                    [df_ratio2, ratio_temp], axis=1)
                df_ratio2 = df_ratio2.rename(
                    columns={0: "ratio_mb/cyt"})
                df_ratio2 = df_ratio2[[
                    "label", "Membrane_signal", "Cytosolic_signal", "ratio_mb/cyt"]]
                df_ratio = pd.concat([df_ratio, df_ratio2], axis=1)

       
    else:
        data.update_console(
            window, "-CONSOLE-", '!!!! Error detected in cell labelling, correction in progress !!!!')
        table_mb2 = table_mb.copy()
        table_cyt2 = table_cyt.copy()
        for wrong_frame in list_frames_tocorrect:
      
            data.update_console(window, "-CONSOLE-",
                           f'correcting frame {str(wrong_frame)}')
            # extract 1st error frame from dataframe
            test = wrong_frame-1
            while test in list_frames_tocorrect:
                test = test-1
            right_frame = test

            correct = table_cyt[(
                table_cyt["Frames"] == right_frame)].copy()
            tocorrect = table_cyt[(
                table_cyt["Frames"] == wrong_frame)].copy()

            correctlabel_list = []
            for y in np.arange(0, len(tocorrect["centroid_x"]), 1):
                dist = []
                for i in np.arange(0, ncells, 1):
                    value = np.sqrt(((tocorrect["centroid_x"][y]-correct["centroid_x"][i])**2)+(
                        (tocorrect["centroid_y"][y]-correct["centroid_y"][i])**2))
                    dist.append(value)
                correctlabel_list.append(dist.index(min(dist))+1)
            mask = table_cyt2["Frames"] == wrong_frame
            table_cyt2.loc[mask, "label"] = correctlabel_list
            mask = table_mb2["Frames"] == wrong_frame
            table_mb2.loc[mask, "label"] = correctlabel_list
        ntf_ref = cytosol_stack_lenght
        tf_ref = list(np.arange(0, ntf_ref, 1))
        labels_ref = np.unique(table_mb2["label"])
        for label in labels_ref:

            ntf = len(table_mb2["Frames"]
                      [table_mb2["label"] == label])
            tf = table_mb2["Frames"][table_mb2["label"]
                                     == label].values
            tf = list(tf)

            if ntf < ntf_ref:
                # find missing ones
                for i in tf_ref:
                    if i not in tf:
                        dummy_row = [
                            label, np.nan, np.nan, np.nan, i]
                        table_mb2.loc[len(table_mb2)] = dummy_row
                        table_cyt2.loc[len(table_cyt2)] = dummy_row

        table_cyt2 = table_cyt2.sort_values(by=['label', "Frames"])
        table_mb2 = table_mb2.sort_values(by=['label', "Frames"])
        
        for label in np.unique(table_cyt2["label"]):
            if label == 1:
                df_cyt = pd.DataFrame(
                    table_cyt2[table_cyt2["label"] == label])
                df_cyt = df_cyt.set_index("Frames")
                df_cyt = df_cyt.drop(
                    ["label", 'centroid_x', 'centroid_y'], axis=1)
                df_cyt = df_cyt.rename(
                    columns={"mean_intensity": "Cytosolic_signal"})

                df_mb = pd.DataFrame(
                    table_mb2[table_mb2["label"] == label])
                df_mb = df_mb.set_index("Frames")
                df_mb = df_mb.drop(
                    ['centroid_x', 'centroid_y'], axis=1)
                df_mb = df_mb.rename(
                    columns={"mean_intensity": "Membrane_signal"})

                df_ratio = pd.concat([df_mb, df_cyt], axis=1)
                ratio_temp = df_ratio["Membrane_signal"].div(
                    df_ratio["Cytosolic_signal"], fill_value=-1)
                df_ratio = pd.concat(
                    [df_ratio, ratio_temp], axis=1)
                df_ratio = df_ratio.rename(
                    columns={0: "ratio_mb/cyt"})
                df_ratio = df_ratio[[
                    "label", "Membrane_signal", "Cytosolic_signal", "ratio_mb/cyt"]]

            else:
                df_cyt = pd.DataFrame(
                    table_cyt2[table_cyt2["label"] == label])
                df_cyt = df_cyt.set_index("Frames")
                df_cyt = df_cyt.drop(
                    ["label", 'centroid_x', 'centroid_y'], axis=1)
                df_cyt = df_cyt.rename(
                    columns={"mean_intensity": "Cytosolic_signal"})

                df_mb = pd.DataFrame(
                    table_mb2[table_mb2["label"] == label])
                df_mb = df_mb.set_index("Frames")
                df_mb = df_mb.drop(
                    ['centroid_x', 'centroid_y'], axis=1)
                df_mb = df_mb.rename(
                    columns={"mean_intensity": "Membrane_signal"})

                df_ratio2 = pd.concat([df_mb, df_cyt], axis=1)
                ratio_temp = df_ratio2["Membrane_signal"].div(
                    df_ratio2["Cytosolic_signal"], fill_value=-1)
                df_ratio2 = pd.concat(
                    [df_ratio2, ratio_temp], axis=1)
                df_ratio2 = df_ratio2.rename(
                    columns={0: "ratio_mb/cyt"})
                df_ratio2 = df_ratio2[[
                    "label", "Membrane_signal", "Cytosolic_signal", "ratio_mb/cyt"]]
                df_ratio = pd.concat([df_ratio, df_ratio2], axis=1)
    return df_ratio
                
def bleach_correction(image_stack, background_intensity=0):
    """
    Perform bleach correction on an image stack using a simple ratio method.

    Parameters:
    image_stack (numpy.ndarray): 3D numpy array representing the image stack. 
                                 Dimensions are [time, height, width].
    background_intensity (float): Intensity value of the background. Defaults to 0.

    Returns:
    numpy.ndarray: 3D numpy array representing bleach-corrected image stack.
    """
    corrected_stack = np.zeros_like(image_stack, dtype=np.float32)
    n_time, height, width = image_stack.shape

    # Calculate the mean intensity of the initial frame (minus background)
    initial_mean_intensity = np.mean(image_stack[0]) - background_intensity

    for t in range(n_time):
        # Calculate the mean intensity of the current frame (minus background)
        current_mean_intensity = np.mean(image_stack[t]) - background_intensity

        # Calculate correction factor
        correction_factor = initial_mean_intensity / \
            current_mean_intensity if current_mean_intensity != 0 else 1

        # Apply correction
        corrected_stack[t] = (image_stack[t] - background_intensity) * \
            correction_factor + background_intensity

    return corrected_stack

import PySimpleGUI as sg
import os
from aicspylibczi import CziFile
import numpy as np
import tifffile
def channel_splitter_czi(window,folder,patternfile):
    """
    Splits channels from .czi files in a specified folder and saves them as .tif files.

    Parameters:
    folder (str): The directory containing .czi files.

    The function performs the following steps:
    1. Searches recursively for .czi files in the specified folder.
    2. Filters files containing 'patternfile' in their name.
    3. Iterates through each .czi file and extracts image data for each channel.
    4. Saves the extracted channel images as .tif files in the same sub-folder.

    Note:
    - The function uses a progress meter for visual feedback.
    - It assumes the presence of specific metadata in the .czi files for channel extraction.
    - The function updates a console element in a GUI window with progress messages.
    """
    czi_files = data.find_file_recursive(folder=folder, pattern=".czi")
    print("found czi files!")
    print(len(czi_files))
    if patternfile != "":
        czi_files = [file for file in czi_files if patternfile in file]
        print("found file with specified pattern")
    else:
        print("no pattern specified, processing all")
        
    if len(czi_files) == 0:
        sg.popup_error('No .czi or no files with pattern found in the specified folder')
    else:
        sg.one_line_progress_meter('Processing Files', 0, len(czi_files), 'Channel splitting')
        for i, file in enumerate(czi_files, 1):  # Start counting from 1
            sub_folder = os.path.dirname(file)
            czi = CziFile(file)
            dimensions = czi.get_dims_shape()
            filename = os.path.basename(file)
            data.update_console(window, "-CONSOLE0-", f"processing {filename}")
            if 'S' in dimensions[0]:#Multi tile files
                print("processing multi tile files")
                for tile in np.arange(dimensions[0]["S"][1]):
                    data.update_console(window, "-CONSOLE0-",
                                   f"opening {filename} position {tile+1}")
                    for channel in np.arange(dimensions[0]["C"][1]):
                        data.update_console(
                            window, "-CONSOLE0-", f"saving {filename} position {tile+1} channel {channel+1}")
                        img, shp = czi.read_image(S=tile, C=channel)
                        img = np.squeeze(img)
                        print(dimensions[0])
                        
                        if dimensions[0]["T"][1]>1 and dimensions[0]["Z"][1]>1:
                            print("dected time and Z")
                            tifffile.imwrite(
                                f"{sub_folder}/C{channel + 1}-{filename}#{tile + 1}.tif",
                                    img, imagej=True, metadata={'axes': 'TZYX'})  # This sets the metadata so that ImageJ knows it's a 3D image)
                        elif dimensions[0]["T"][1]>1 and dimensions[0]["Z"][1]==1:
                            print("dected time ")
                            tifffile.imwrite(
                                f"{sub_folder}/C{channel + 1}-{filename}#{tile + 1}.tif",
                                    img, imagej=True, metadata={'axes': 'TYX'}) 
                        elif dimensions[0]["T"][1]==1 and dimensions[0]["Z"][1]>1:
                            print("dected Z ")
                            tifffile.imwrite(
                                f"{sub_folder}/C{channel + 1}-{filename}#{tile + 1}.tif",
                                    img, imagej=True, metadata={'axes': 'ZYX'})  
                        elif dimensions[0]["T"][1]==1 and dimensions[0]["Z"][1]==1:
                            print("no time no z")
                            tifffile.imwrite(
                                f"{sub_folder}/C{channel + 1}-{filename}#{tile + 1}.tif",
                                    img, imagej=True, metadata={'axes': 'YX'})
                        
                            
            else:#Single tiles files
                print("processing single tile files")
                for channel in np.arange(dimensions[0]["C"][1]):
                    data.update_console(
                        window, "-CONSOLE0-", f"saving {filename} position channel {channel+1}")
                    img, shp = czi.read_image(C=channel)
                    img = np.squeeze(img)
                    
                    if dimensions[0]["T"][1]>1 and dimensions[0]["Z"][1]>1:
                        tifffile.imwrite(
                            f"{sub_folder}/C{channel + 1}-{filename}.tif",
                                img, imagej=True, metadata={'axes': 'TZYX'})  # This sets the metadata so that ImageJ knows it's a 3D image)
                    elif dimensions[0]["T"][1]>1 and dimensions[0]["Z"][1]==1:
                        tifffile.imwrite(
                            f"{sub_folder}/C{channel + 1}-{filename}.tif",
                                img, imagej=True, metadata={'axes': 'TYX'}) 
                    elif dimensions[0]["T"][1]==1 and dimensions[0]["Z"][1]>1:
                        tifffile.imwrite(
                            f"{sub_folder}/C{channel + 1}-{filename}.tif",
                                img, imagej=True, metadata={'axes': 'ZYX'})
                    elif dimensions[0]["T"][1]==1 and dimensions[0]["Z"][1]==1:
                        tifffile.imwrite(
                            f"{sub_folder}/C{channel + 1}-{filename}.tif",
                                img, imagej=True, metadata={'axes': 'YX'})
                        
            # Update the progress meter
            if not sg.one_line_progress_meter('Processing Files', i, len(czi_files), 'Channel splitting'):
                print('User cancelled')
                break
                
            

import datetime
import subprocess
import locale
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
def registration(referenceChannelFiles, applyToFiles,ij_path,macro_path):
    """
    Performs image registration using ImageJ's macro and subprocess.

    Parameters:
    referenceChannelFiles (list): A list of file paths for reference channel files.
    applyToFiles (list): A list of file paths to which the registration is to be applied.

    The function constructs a command to run an ImageJ macro with specified parameters for image registration and executes it using subprocess.

    Note:
    - The function defines several parameters for the registration process.
    - It constructs a command string to run ImageJ with the specified macro and parameters.
    - The function uses subprocess to execute the command.
    - ImageJ is forcibly closed after the registration process.
    """
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    # Define the parameters in Python
    exp_nro = "001"
    XY_registration = "true"
    projection_type_xy = "Average Intensity"
    time_xy = "1"
    max_xy = "0"
    reference_xy = "previous frame (better for live)"
    crop_output = "true"
    z_registration = "true"
    projection_type_z = "Average Intensity"
    reslice_mode = "Left"
    time_z = "1"
    reference_z = "first frame (default, better for fixed)"
    extend_stack_to_fit = "true"
    ram_conservative_mode = "false"
    max_z = "0"
    files = ",".join(referenceChannelFiles)
    files2 = ",".join(applyToFiles)
    current_date = datetime.date.today()
    date= current_date.strftime('%Y-%b-%e')
    date= date.replace(" ", "")
    # Combine them into a single string separated by semicolons

    params = ";".join([exp_nro, files, XY_registration, projection_type_xy, time_xy, max_xy,
                       reference_xy, crop_output, z_registration, projection_type_z, reslice_mode,
                       time_z, reference_z, extend_stack_to_fit, ram_conservative_mode, max_z,
                       files2, date])
    # Construct the ImageJ macro call
    cmd = f"{ij_path} -macro {macro_path} \"{params}\""

    # Run the command using subprocess
    subprocess.run(cmd, shell=True)
    subprocess.run(["pkill", "ImageJ-linux64",])
    
def corner_average(image):
    """
    Calculates the average pixel value of the corners of an image, excluding outliers.

    Parameters:
    image (array): The image array to process.

    The function performs the following steps:
    1. Extracts the corner regions of the image.
    2. Calculates the average pixel value for each corner.
    3. Discards corner averages that significantly deviate from the median.
    4. Returns the overall average of the valid corner averages.

    Returns:
    float: The overall average of the corner pixel values, or NaN if all are discarded as outliers.

    Note:
    - The function is designed to exclude corners with atypical brightness, possibly due to artifacts.
    """
    size=int(min(image.shape)*0.07)
    corners = [
        image[0:size, 0:size],  # top_left
        image[0:size, -size:],  # top_right
        image[-size:, 0:size],  # bottom_left
        image[-size:, -size:]  # bottom_right
    ]

    # Calculate averages for each corner
    corner_avgs = [np.mean(corner) for corner in corners]

    # Calculate the median of the corner averages
    median_avg = np.median(corner_avgs)

    # Discard corner averages that are 10% different from the median
    valid_avgs = [avg for avg in corner_avgs if abs(
        avg - median_avg) / median_avg <= 1]

    # Calculate and return the overall corner average using valid averages
    if valid_avgs:
        overall_avg = np.mean(valid_avgs)
        return overall_avg
    else:
        return np.nan  # return nan if all averages are discarded
    
from utils import image_analyses, data, plot
import pandas as pd
import matplotlib.pyplot as plt

def tracerAnalysis(file,path):
    data_im = file  # import full stabilized C1 file
    
    mip_data = np.mean(data_im, axis=1)  # mean Z projection on valid frames only
    mip_data=np.ma.masked_equal(mip_data, 0)
    df = pd.DataFrame(
        {'Timeframe': range(len(mip_data)), 'Corner Average': np.mean(mip_data,axis=(1,2))})
    print("mip_data")
    print(mip_data)
    # Calculate the first-order difference
    diffs = np.diff(np.mean(mip_data,axis=(1,2)))
    print("diffs")
    print(diffs)
    # Compute mean and standard deviation of differences
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # Identify significant changes (more than 2 std_devs from the mean)
    threshold = mean_diff + 3 * std_diff
    significant_changes=[]
    significant_changes = np.where(np.abs(diffs) > threshold)[0]
    if significant_changes:
        image_parameters = [f'Number of timeframes:{data_im.shape[0]}',
                            f'Number of slices:{data_im.shape[1]}',
                            f'Dimension x:{data_im.shape[2]}',
                            f'Dimension y:{data_im.shape[3]}',
                            f'Tracer significant change at frame:{significant_changes[0]}']
    else:
        image_parameters = [f'Number of timeframes:{data_im.shape[0]}',
                            f'Number of slices:{data_im.shape[1]}',
                            f'Dimension x:{data_im.shape[2]}',
                            f'Dimension y:{data_im.shape[3]}',
                            'Tracer significant change at frame:no significant change found']
        
    # Split each string into name and value
    split_parameters = [param.split(":") for param in image_parameters]

    # Create DataFrames and save
    df.to_csv(f'{path}_tracer_measured.csv')
    df_im = pd.DataFrame(split_parameters, columns=[
                         "Parameter", "Value"])
    df_im.to_csv(f'{path}_parameters.csv')

    # Save tracer plot
    # Create the figure and axes objects
    tracer_figure, ax = plt.subplots(figsize=(10, 6))

    # Create the plot on the Axes object
    ax.plot(np.mean(mip_data,axis=(1,2)), label='Averages')

    # Add significant changes
    if significant_changes:
        for change in significant_changes:
            ax.axvline(change, color='g', linestyle='--',
                       label=f'Significant Change at {change}')
    else:
            ax.axvline(0, color='g', linestyle='--',
                       label="No significant changes found")

    # Add labels and title
    ax.set_xlabel('Timeframe')
    ax.set_ylabel('Average')
    ax.set_title(' Averages Over Time with Significant Changes')

    # Add legend
    ax.legend()

    # Save the plot
    tracer_figure.savefig(f'{path}_tracer_averages_plot.png')     

def parameterfile(file,path):
    data_im = file  # import full stabilized C1 file
    # Filter out fully black frames before calculating mean Z projection
    image_parameters = [f'Number of timeframes:{data_im.shape[0]}',
                        f'Number of slices:{data_im.shape[1]}',
                        f'Dimension x:{data_im.shape[2]}',
                        f'Dimension y:{data_im.shape[3]}',
                        'Tracer significant change at frame:na']
    # Split each string into name and value
    split_parameters = [param.split(":") for param in image_parameters]
    df_im = pd.DataFrame(split_parameters, columns=[
                         "Parameter", "Value"])
    df_im.to_csv(f'{path}_parameters.csv')

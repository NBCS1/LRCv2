import os
from utils.data import filenamesFromPaths,find_file
import tifffile
import re
from utils.plot import testDisplay
import numpy as np
from napari.qt import thread_worker
import napari
from utils.data import createTestJson
# MagicGUI
from magicgui import widgets
from magicgui import magicgui
import warnings
import numpy as np
from matplotlib.path import Path
from utils import data
import os
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication 
def napariTestFunction(img_pi,img_fluo,app,imagesPath):
    # Create a Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(img_fluo,name="Biosensor")
    viewer.add_image(img_pi,name="Plasma membrane ref")
    # Define a function to update the viewer with the slider values
    @thread_worker(connect={
        'medianSlider': 'value',
        'maxSlider': 'value',
        'tophatSlider': 'value',
        'cr1Slider': 'value',
        'cr2Slider': 'value',
        'dilate1Slider': 'value',
        'dilate2Slider': 'value',
        'erosionSlider': 'value',
        'vminSlider': 'value',
        'vmedianSlider': 'value',
        'biomedianSlider': 'value',
        'biotophatSlider': 'value',
        'erosionfactorSlider': 'value',
        'thresholdType': 'value',
        'processorNot': 'checked'
    })
    def update_viewer(**kwargs):
        filters_info = ', '.join([f"{key}: {value}" for key, value in kwargs.items()])
        viewer.text_overlay.text = filters_info
    
    # Create a magicgui widget
    @magicgui(call_button='Apply Filters',thresholdType={"choices": ["Otsu Threshold", "Median Threshold"]})
    def filter_widget(
            img_pi,img_fluo,imagesPath,
            medianSlider: int = 10,
            maxSlider: int = 5,
            tophatSlider: int = 20,
            cr1Slider: int = 2,
            cr2Slider: int = 10,
            dilate1Slider: int = 15,
            dilate2Slider: int = 3,
            erosionSlider: int = 15,
    
            thresholdType: str = "Otsu Threshold",
            vminSlider: int = 5,
            vmedianSlider: int = 2,
    
            processorNot: bool = False,
            biomedianSlider: int = 2,
            biotophatSlider: int = 20,
            erosionfactorSlider: int = 1
    ):    
            apply_filters(
            medianSlider,
            maxSlider,
            tophatSlider,
            cr1Slider,
            cr2Slider,
            dilate1Slider,
            dilate2Slider,
            erosionSlider,
            thresholdType,
            vminSlider,
            vmedianSlider,
            processorNot,
            biomedianSlider,
            biotophatSlider,
            erosionfactorSlider
        )
    def apply_filters(
        median_slider,
        max_slider,
        tophat_slider,
        cr1_slider,
        cr2_slider,
        dilate1_slider,
        dilate2_slider,
        erosion_slider,
        threshold_type,
        vmin_slider,
        vmedian_slider,
        processor_not,
        biomedian_slider,
        biotophat_slider,
        erosionfactor_slider,
        ):

        testParams={
                        'median_radius': median_slider,
                        'max_filter_size': max_slider,
                        'top_hat_radius': tophat_slider,
                        'closing_radius1': cr1_slider,
                        'closing_radius2': cr2_slider,
                        'dilation_radius1': dilate1_slider,
                        'dilation_radius2': dilate2_slider,
                        'thresholdtype': threshold_type,
                        'vmin': vmin_slider,
                        'vmedian': vmedian_slider,
                        'biomedian': biomedian_slider,
                        'biotophat': biotophat_slider,
                        'dontprocess':  processor_not,
                        'erosion_radius':erosionfactor_slider,
                    }
        membranes1, cytcorrected1, endosomes1=segmentation_all(image_pi=img_pi, image_biosensor=img_fluo, erosionfactor= erosion_slider, params=testParams)
        endosomes1=endosomes1.astype(int)
        membranes1=membranes1.astype(int)
        viewer.add_labels(membranes1,name="Segm. PM")
        viewer.add_labels(endosomes1,name="Segm. Cyt")
        
        pathtest=os.path.dirname(imagesPath)
        createTestJson(testParams=testParams,path=pathtest+'/custom_config.json')
        
    # Add the magicgui widget to Napari
    filter_widget.img_pi.bind(img_pi)
    filter_widget.img_fluo.bind(img_fluo)
    filter_widget.imagesPath.bind(imagesPath)
    viewer.window.add_dock_widget(filter_widget, area='right')
    
    # Show the viewer
    viewer.show()
    app.exec_()

def testImage(imagesPath,window,app):
    """Function to test and found the margin for filters values to apply to a set of data
    Parameters:
        imageC1: tif image of the biosensor
        imageC2: tif image of the PI or membrane staining
    Return a multi columns image presenting 4 different set of filter values for both images
    Save it in the dedicated folder
    """
    #Import from path
    path=os.path.dirname(imagesPath)#found path
    imageName=os.path.basename(imagesPath)#found filename
    imageNameNoChannel=imageName[3:]
    tifpositions=re.search(".tif",imageNameNoChannel)
    postion1=tifpositions.span()[0]
    imageNameNoChannel=imageNameNoChannel[:postion1+4]
    #Retrieve filepath of C1 and C2
    list_files=find_file(folder=path,pattern=imageNameNoChannel)
    list_pattern_avoid=[]
    imgpathC1,imgpathC2=filenamesFromPaths(list_files,list_pattern_avoid)
    
    #Open C1 and C2
    imgC1 = tifffile.imread(imgpathC1,key=0)
    imgC2 = tifffile.imread(imgpathC2,key=0)
    #Set 1 gentle for beautiful PI staining
    
    napariTestFunction(imgC2,imgC1,app,imagesPath)
    
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
    
    # Apply segmentation with varying parameters
    #Denoising, preserving edges
    denoised_image = cle.median_box(image_pi, radius_x=params["median_radius"], radius_y=params["median_radius"], radius_z=0)
    
    #Feature/edge enhancement
    denoised_image2 = ndimage.maximum_filter(cle.pull(denoised_image), size=params["max_filter_size"])
    
    # Top hat background correction>highlight small and bright features
    denoised_image2 = cle.top_hat_box(
        denoised_image2, radius_x=params["top_hat_radius"], radius_y=params["top_hat_radius"], radius_z=0)
    
    # Square root box filter to enhance low contrast images
    denoised_image3 = cle.sqrt(denoised_image2)
    
    # Otsu auto threshold
    binary1 = cle.threshold_otsu(denoised_image3)
    
    # Closing operation to fill gaps
    binary = cle.closing_labels(binary1, radius=params["closing_radius1"])
    skeleton = skeletonize(cle.pull(binary))
    skeleton = (skeleton > 0).astype(np.uint8) * 255
    pruned_skeleton, segmented_img, segment_objects = prune(
        skel_img=skeleton, size=1000)
    cle_image = cle.push(pruned_skeleton)
    dilate = cle.dilate_labels(cle_image, cle_image, radius=params["dilation_radius1"])
    dilate = cle.closing_labels(dilate, radius=params["closing_radius2"])
    dilate = cle.erode_labels(dilate, radius=params["erosion_radius"])
    dilate = skeletonize(cle.pull(dilate))
    cle_image = cle.push(dilate)
    dilate = cle.dilate_labels(cle_image, cle_image, radius=params["dilation_radius2"])
    dilate = cle.closing_labels(dilate, radius=params["closing_radius2"])
    inverted = np.asarray(dilate) == 0 * 1
    label = cle.connected_components_labeling_box(inverted)
    exclude = cle.exclude_labels_on_edges(label)
    exclude = cle.exclude_labels_outside_size_range(
        exclude, minimum_size=2500, maximum_size=100000000)

    # vacuole removal
    denoised_image = cle.median_box(
        image_biosensor, radius_x=params["vmedian"], radius_y=params["vmedian"], radius_z=0)
    mini = cle.minimum_box(denoised_image, radius_x=params["vmin"], radius_y=params["vmin"], radius_z=0)
    if params['thresholdtype']=="Otsu Threshold":
        binary2 = cle.threshold_otsu(mini)
    else:
        binary2=cle.threshold(mini,constant=np.median(mini))
        
    inverted2 = np.asarray(binary2) == 0 * 1
    cytcorrected = cle.binary_subtract(exclude, inverted2)

    # remove out of range label
    extend = cle.extend_labels_with_maximum_radius(exclude, radius=7)
    membranes = cle.binary_subtract(extend, label)
    
    # Keep endosomes from cytosolic signal
    denoised_image2 = cle.median_box(
        image_biosensor, radius_x=params['biomedian'], radius_y=params['biomedian'])
    cyt_one = cle.divide_images(cytcorrected, cytcorrected)
    # 5 for 564, 3 for 604? and 1 for 991
    cyt_one = cle.erode_labels(cyt_one, radius=erosionfactor)
    cyt_one = cle.multiply_images(cyt_one, denoised_image2)
    if params['dontprocess']=="true":
        denoised_image3=cyt_one
    else:
        denoised_image3 = cle.top_hat_box(
            cyt_one, radius_x=params["biotophat"], radius_y=params["biotophat"])
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
        # Apply segmentation with varying parameters
        #Denoising, preserving edges
        denoised_image = cle.median_box(frame, radius_x=params["median_radius"], radius_y=params["median_radius"], radius_z=0)
        
        #Feature/edge enhancement
        denoised_image2 = ndimage.maximum_filter(cle.pull(denoised_image), size=params["max_filter_size"])
        
        # Top hat background correction>highlight small and bright features
        denoised_image2 = cle.top_hat_box(
            denoised_image2, radius_x=params["top_hat_radius"], radius_y=params["top_hat_radius"], radius_z=0)
        
        # Square root box filter to enhance low contrast images
        denoised_image3 = cle.sqrt(denoised_image2)
        
        # Otsu auto threshold
        binary1 = cle.threshold_otsu(denoised_image3)
        
        # Closing operation to fill gaps
        binary = cle.closing_labels(binary1, radius=params["closing_radius1"])
        skeleton = skeletonize(cle.pull(binary))
        skeleton = (skeleton > 0).astype(np.uint8) * 255
        pruned_skeleton, segmented_img, segment_objects = prune(
            skel_img=skeleton, size=1000)
        cle_image = cle.push(pruned_skeleton)
        dilate = cle.dilate_labels(cle_image, cle_image, radius=params["dilation_radius1"])
        dilate = cle.closing_labels(dilate, radius=params["closing_radius2"])
        dilate = cle.erode_labels(dilate, radius=params["erosion_radius"])
        dilate = skeletonize(cle.pull(dilate))
        cle_image = cle.push(dilate)
        dilate = cle.dilate_labels(cle_image, cle_image, radius=params["dilation_radius2"])
        dilate = cle.closing_labels(dilate, radius=params["closing_radius2"])
        skeleton_stack.append(cle.pull(dilate))
        inverted = np.asarray(dilate) == 0 * 1
        label = cle.connected_components_labeling_box(inverted)
        exclude = cle.exclude_labels_on_edges(label)
        exclude = cle.exclude_labels_outside_size_range(
            exclude, minimum_size=2500, maximum_size=100000000)
        
        # vacuole removal
        denoised_image = cle.median_box(
            frameps, radius_x=params["vmedian"], radius_y=params["vmedian"], radius_z=0)
        mini = cle.minimum_box(denoised_image, radius_x=params["vmin"], radius_y=params["vmin"], radius_z=0)
        if params['thresholdtype']=="Otsu Threshold":
            binary2 = cle.threshold_otsu(mini)
        else:
            binary2=cle.threshold(mini,constant=np.median(mini))
            
        inverted2 = np.asarray(binary2) == 0 * 1
        cytcorrected = cle.binary_subtract(exclude, inverted2)
        cytosol_stack.append(cle.pull(cytcorrected))
        
        # remove out of range label
        extend = cle.extend_labels_with_maximum_radius(exclude, radius=7)
        membranes = cle.binary_subtract(extend, label)
        membrane_stack.append(membranes)
        # Keep endosomes from cytosolic signal
        denoised_image2 = cle.median_box(
            frameps, radius_x=params['biomedian'], radius_y=params['biomedian'])
        cyt_one = cle.divide_images(cytcorrected, cytcorrected)
        # 5 for 564, 3 for 604? and 1 for 991
        cyt_one = cle.erode_labels(cyt_one, radius=erosionfactor)
        cyt_one = cle.multiply_images(cyt_one, denoised_image2)
        if params['dontprocess']=="true":
            denoised_image3=cyt_one
        else:
            denoised_image3 = cle.top_hat_box(
                cyt_one, radius_x=params["biotophat"], radius_y=params["biotophat"])
        endosomes = cle.threshold_otsu(denoised_image3)#denoised image 3
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

from sklearn.cluster import KMeans
def autoCorrectLabels(df):
    # Prepare the data for clustering
    data = df[['centroid_x', 'centroid_y']].values
    nCellMax=int(df["label"].max())
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=nCellMax, random_state=0).fit(data)
    df['cluster'] = kmeans.labels_
    
    # Calculate the average centroids for each cluster
    average_centroids = df.groupby('cluster').agg({
        'centroid_x': 'mean',
        'centroid_y': 'mean'
    }).reset_index()
    
    # Sort clusters based on the average y-coordinate in descending order
    average_centroids = average_centroids.sort_values(by='centroid_y', ascending=True).reset_index(drop=True)
    average_centroids['new_label'] = range(1, len(average_centroids) + 1)
    
    # Create a mapping from the old cluster labels to the new labels
    label_mapping = {row['cluster']: row['new_label'] for _, row in average_centroids.iterrows()}
    df['new_label'] = df['cluster'].map(label_mapping)
    
    #save ref of exhanges
    ref=df[["label","new_label"]].copy()
    # Get unique labels
    df['label']=df["new_label"]
    df.drop(columns=["cluster","new_label"],inplace=True)
    
    unique_labels = df['label'].unique()
    for frame in df['Frames'].unique():
        # Get labels detected in this frame
        labels_detected = df[df['Frames'] == frame]['label'].unique()
        
        # Check for missing labels
        missing_labels = np.setdiff1d(unique_labels, labels_detected)
        
        # Add dummy rows for missing labels
        for label in missing_labels:
            df = df.append({'Frames': frame, 'label': label, 'centroid_x': np.nan, 'centroid_y': np.nan,"mean_intensity":np.nan}, ignore_index=True)
    df = df.sort_values(by=['label','Frames'])
    df = df.reset_index(drop=True)
    return df,ref

def applyLabelCorrectionToCytosol(ref,df):
    df["label"]=ref["new_label"]
    unique_labels = df['label'].unique()
    for frame in df['Frames'].unique():
        # Get labels detected in this frame
        labels_detected = df[df['Frames'] == frame]['label'].unique()
        
        # Check for missing labels
        missing_labels = np.setdiff1d(unique_labels, labels_detected)
        
        # Add dummy rows for missing labels
        for label in missing_labels:
            df = df.append({'Frames': frame, 'label': label, 'centroid_x': np.nan, 'centroid_y': np.nan,"mean_intensity":np.nan}, ignore_index=True)
    df = df.sort_values(by=['label','Frames'])
    df = df.reset_index(drop=True)
    return df
    
                
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

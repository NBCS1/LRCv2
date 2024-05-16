'''
author: Nelson BC Serre, SICE team, RDP, ENS de Lyon. 2024

The LRC software provide a graphical interface to test and then automatically
segment Arabidopsis thaliana plasma membrane and cytosolic fluorescence signal.
The software works with movies and single frames and assist the user in ROI
selection.
The software segments plasma membranes from a reference channel
    e.g.: Propidium iodide, lti6b-fluorophore
And then measure the plasma membrane and cytosolic signal to calculate the
ratio in order to quantify the subcellular dynamics of a given fluorophore.
The software requires .czi files or individual .tif images/stack with 
individual channels.

Steps:
    Channel splitting
    Stabilisation in Z and XY
    Semi-automated ROI selection
    Segmentation
    Quantification
    Compilation and plotting

main.py : Handles the events triggered by the GUI
Depends on utils.image_analyses.py, utils.data.py, utils.plot.py, napari_roi.py
and main_window.py

--------------------------To do list--------------------------
Export all the values in compile (add raw compile, do not deleted the mean compile as LRC needs it to plot)

No signal, set to 0 not NA and no division just set ratio to 0

Progress bar during single frame and movie ROI selection

Save nparcomp summary as text file

--------------------------Bug to fix--------------------------

after relabel actually relabel the images

handle extra cells>ignore new, zero cell >NA values or new cell add empty retroactively?

Plots in Windows are completly crushed, set a default canvas size and plot size??

'''

# Standard Library Imports
import os
import sys

import subprocess
import datetime
import warnings
from datetime import date

# Third-Party General Purpose Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import tifffile

# GUI Libraries
import PySimpleGUI as sg


# Image Processing and Scientific Libraries

import napari
from skimage.io import imsave
from skimage.morphology import skeletonize
#LRC utilities
from gui.main_window import launch_main_gui,open_parameter_popup
from gui.napari_roi import napariROI, napariROI_single, napariTracer
from utils import image_analyses, data, plot
from plantcv.plantcv.morphology import prune
import pyclesperanto_prototype as cle
from utils.image_analyses import testImage

#initialization of program dependencies paths
import json

with open('config.json', 'r') as file:
    config = json.load(file)
    print(config)
#Image-J interface
ij_path = config["ij_path"]

# R Interface Libraries
os.environ['R_HOME'] = config["R_HOME"]
os.environ["R_LIBS"] = config["R_LIBS"]

#option parameters
params=config["parameters"]
print(params)
gpu=config["selected_gpu"]

import rpy2.rinterface as rinterface
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, Formula

lrc_directory=os.getcwd()

#Retrieve date in different formats
current_date = datetime.date.today()
date_str = current_date.strftime("%Y %B %d")
day = current_date.strftime('%Y-%b-%d')


version = 5
global firstfile
firstfile = True
compiled_data = None
file_list = []  # A list to keep track of added files
plotcompiled = None

from PyQt5.QtWidgets import QApplication  

app = QApplication(sys.argv) 
# global env variables initialization (related to the single frames statistics)
stat_export = False
d1 = None
test_result = None
cld_result = None

def main(stat_export=stat_export):
    window = launch_main_gui(name="LRC: Lipid Ratio calculator v1 beta")

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Cancel"):#Close window button event
            window.close()
            break

        if event == "Split":#Split button event
            if values["-FOLDER0-"] != "":
                image_analyses.channel_splitter_czi(window,values["-FOLDER0-"],patternfile=values['patternsplit'])
            else:
                sg.popup_error('Please select a folder first')
                
        if event == "3D-Registration":#3D registration button even  
            ij_path,params,gpu=data.readParameters()
            cle.select_device(gpu)
            if values["-FOLDER00-"] != "":
                folder = values["-FOLDER00-"]
                list_files = data.find_file_recursive(folder=folder, pattern=".tif")  # look for tif files only
                list_pattern_avoid = ["Drift-plot", "xyCorrected", "xyzCorrected"]
                list_fileC1,list_fileC2 = data.filenamesFromPaths(list_files,list_pattern_avoid)
                if len(list_fileC2) == 0:
                    sg.popup_error(
                        'Not C2 channel files detected, run SPLIT first')
                elif len(list_fileC1) == 0:
                    sg.popup_error(
                        'Not C1 channel files detected, run SPLIT first')
                else:
                    image_analyses.registration(referenceChannelFiles=sorted(
                        list_fileC2), applyToFiles=sorted(list_fileC1),ij_path=ij_path,macro_path="fast4Dregheadlessv1.ijm")
            else:
                sg.popup_error('Please select a folder first')
                
                
        if event == "Manual ROI selection":#Launch Napari viewer for manual ROI selection
            if values["-FOLDER0000-"] != "":
                folder = values["-FOLDER0000-"]
                list_files = data.find_file_recursive(folder=folder, pattern=".tif")
                list_files = data.find_file_recursive(folder=folder, pattern="xyzCorrected.tif")# look for tif files only
                # extract filenames from full path
                list_pattern_avoid = ["Drift-plot", "xyCorrected"]
                list_fileC1,list_fileC2 = data.filenamesFromPaths(list_files,list_pattern_avoid)
                if len(list_fileC1) == 0:
                    sg.popup_error(
                        'No C1 channel xyzCorrected.tif files detected, run 3D registration first')
                elif len(list_fileC2)==0:
                    sg.popup_error(
                        'No C2 channel xyzCorrected.tif files detected, run 3D registration first')
                else:
                    napariROI(list_fileC2=sorted(list_fileC2), list_fileC1=sorted(list_fileC1),app=app)
                    #Check if tracer analysis roi selection was ticked
                    if values["analysetracer"]:
                        napariTracer(sorted(list_fileC1),app)

            else:
                sg.popup_error('Please select a folder first')
        

                
        if event == "Run image processing":###Start movie segmentation
            ij_path,params,gpu=data.readParameters()
            cle.select_device(gpu)
            #process user input
            if values["erosion"] == "564":
                erosionfactor = 5
            elif values["erosion"] == "991":
                erosionfactor = 1
            elif values["erosion"] == "604":
                erosionfactor = 3

            nb_folders_toanalyse=data.folderToAnalyze(values=values)# determine how many folder were specified by user
           
            try:
                for folder_nb in np.arange(1, nb_folders_toanalyse+1, 1):
                    folder_path = values["-FOLDER"+str(folder_nb)+"-"]
                    savename = os.path.basename(folder_path)
                    data.update_console(
                        window, "-CONSOLE-", f'Analysing folder number {str(folder_nb)}/{str(nb_folders_toanalyse)}')

                    table_mb,table_cyt,cytosol_stack=image_analyses.segmentationMovie( directory = folder_path,
                                                                                      window=window,
                                                                                      erosionfactor=erosionfactor,
                                                                                      values=values,params=params)
                    os.chdir(lrc_directory)
                    
                    table_mb_corrected,ref=image_analyses.autoCorrectLabels(df=table_mb)

                    table_cyt_corrected=image_analyses.applyLabelCorrectionToCytosol(ref,table_cyt)

                    df_ratio=data.movieRatios(table_cyt_corrected,table_mb_corrected,window,values)

                    #df_ratio.to_csv(folder_path+"test.csv")
                    data.adjustTimeTracer(dataframe=df_ratio,
                                          folder_path=folder_path,
                                          version=version,
                                          erosionfactor=erosionfactor,
                                          date_str=date_str,
                                          savename=savename)
                    
            except Exception as e:
                data.update_console(
                    window, "-CONSOLE-", f"An error occurred while processing folder number {folder_nb}: {str(e)}")
                data.update_console(window, "-CONSOLE-",
                               "Continuing with the next folder.")
                os.chdir(lrc_directory)

            data.update_console(window, "-CONSOLE-", "DONE!!")
            sg.popup_no_frame('Image analysis is done!')
            os.chdir(lrc_directory)
        
        if event =="Split Channels":
            print("splitting channels")
            rawFolder=values["-FOLDER1122-"]#retrieve folder specified by user
            #check if the folder was specified by user
            if rawFolder == "":
                sg.popup_error(
                    "No folder was specified", title="Folder error")
            
            image_analyses.channel_splitter_czi(window,rawFolder,patternfile=values['patternsplit'])
            
            
            
        if event == "Run image processing Single frame":
            ij_path,params,gpu=data.readParameters()#read parameters in config file
            cle.select_device(gpu)#activate selected gpu
            
            #retrieve the erosion factor
            if values["erosion"] == "564":
                erosionfactor = 5
            elif values["erosion"] == "991":
                erosionfactor = 1
            elif values["erosion"] == "604":
                erosionfactor = 3
            
            #retrieve splitted image folder
            biosensor_folder = values["-FOLDER12-"]
            pi_folder = biosensor_folder
            
            #check if the folder was specified by user
            if biosensor_folder == "":
                sg.popup_error(
                    "No folder was specified", title="Folder error")

            # retrieve image list C1 > biosensor
            biosensor_img = data.find_file(folder=biosensor_folder, pattern="C1")
            biosensor_img.sort()
            
            # retrieve image list C2 > Pi
            pi_img = data.find_file(folder=pi_folder, pattern="C2")
            pi_img.sort()
            
            # check same number of images
            if len(biosensor_img) != len(pi_img):
                sg.popup_error(
                    "Images for biosensor and pi are not matching in numbers", title="file error")
                
            #Check if image have the same name except for the C1 or C2 channel number
            biosensor_img_base = [os.path.basename(file) for file in biosensor_img]#retrieve images filenames
            biosensor_img_base = [file.replace(
                file[0:3], "") for file in biosensor_img_base]#Remove C1 or C2 in front
            biosensor_img_base.sort()#sort image
            pi_img_base = [os.path.basename(file) for file in pi_img]#Same for PI images
            pi_img_base = [file.replace(file[0:3], "") for file in pi_img_base]
            pi_img_base.sort()
            output_compare = data.compareList(l1=biosensor_img_base, l2=pi_img_base)#Compare if sorted list have the same names
            if output_compare == "Non equal":
                sg.popup_error("Image names are not matching", title="file error")

            #ROI selection with NAPARI from the two list
            napariROI_single(pi_img,biosensor_img,app)
            
            #Retrieve processed images from Napari new folder processed
            processed_folder=biosensor_folder+"/processed/"
            biosensor_img = data.find_file(folder=processed_folder, pattern="fluo.tif")
            biosensor_img.sort()
            pi_img = data.find_file(folder=processed_folder, pattern="pi.tif")
            pi_img.sort()
            if len(biosensor_img) != len(pi_img):
                sg.popup_error(
                    "Images for biosensor and pi are not processed?", title="file error")
            print("Same number of fluo and pi images, next")

            #Check if there is files for both pi and fluo and they have the same name except for the channel and -fluo.tif pi.tif
            biosensor_img_base = [os.path.basename(file) for file in biosensor_img]
            biosensor_img_base = [file.replace(
                file[0:3], "") for file in biosensor_img_base]
            biosensor_img_base=[file.replace(
               file[len(file)-9:], "") for file in biosensor_img_base]
            biosensor_img_base.sort()
            pi_img_base = [os.path.basename(file) for file in pi_img]
            pi_img_base = [file.replace(file[0:3], "") for file in pi_img_base]
            pi_img_base = [file.replace(file[len(file)-7:], "") for file in pi_img_base]
            pi_img_base.sort()
            
            output_compare = data.compareList(l1=biosensor_img_base, l2=pi_img_base)
            if output_compare == "Non equal":
                sg.popup_error("Image names are not matching", title="file error")
            
            print("Binomes of fluo and pi images are good to go!")
            
            # process each binome of file one by one
            i = 0
            try:
                print("processing images")
                for img_pi_path, img_biosensor_path in zip(pi_img,biosensor_img):
                    img_nb = len(biosensor_img)
                    data.update_console(
                        window, "-CONSOLE1-", f'Analysing couple of images number {i+1}/{img_nb}')
                    image_pi = tifffile.imread(img_pi_path)
                    image_pi=image_pi.squeeze()
                    image_biosensor = tifffile.imread(img_biosensor_path)
                    image_biosensor=image_biosensor.squeeze()
                    data.update_console(window, "-CONSOLE1-",'generating segmentation')
                    print("Images are loaded")
                    print(image_pi.shape)
                    print(image_biosensor.shape)
                    membranes, novacuole, intracellular = image_analyses.segmentation_all(image_pi=image_pi,
                                                                                          image_biosensor=image_biosensor,
                                                                                          params=params,
                                                                                          erosionfactor=erosionfactor)
                
                    plot.singleFrameAnalysisDisplay(membranes=membranes,
                                               novacuole=novacuole,
                                               intracellular=intracellular,
                                               window=window)
 
                    data.singleFrameDataMeasures(image_biosensor=image_biosensor,
                                                 membranes=membranes,
                                                 intracellular=intracellular,
                                                 img_pi_path=img_pi_path,
                                                 img_biosensor_path=img_biosensor_path,
                                                 day=day)
                    i += 1
            except Exception as e:
                data.update_console(
                    window, "-CONSOLE1-", f"An error occurred while processing folder number {img_biosensor_path}: {str(e)}")
                data.update_console(window, "-CONSOLE1-",
                               "Continuing with the next folder.")

            sg.popup_no_frame('Image analysis is done!')
        
        if event in ("Options1", "Options2"):
            open_parameter_popup()

            
        if event == 'Clear':
            window['-FOLDER1-'].update(value='')
            window['-FOLDER2-'].update(value='')
            window['-FOLDER3-'].update(value='')
            window['-FOLDER4-'].update(value='')
        
        if event == "Compile":
            #recreate compile data here instead of with ADD and then process the data, change function for process to
            #retrieve folder
            folder_name = values['File']
            #retrieve "csv" files
            files_list=data.find_file_recursive(folder_name, ".csv")
            #get only "results-LRC
            results_list=data.search_pattern_recursive(files_list, "results-LRC")
            if len(results_list)==0:
                results_list=data.search_pattern_recursive(files_list, "_analysis.csv")
                singleprocessFlag=True
            else:
                singleprocessFlag=False
            compiled_data=[]
            datas=[]
            for file_name in results_list:
                if singleprocessFlag==False:  # movie processesing
                    try:
                        timeframeDuration= float(values["timeframeDuration"])
                        if timeframeDuration==0:
                            timeframeDuration=1
                    except ValueError:
                        sg.popup('Please enter a valid timeframe interval')
                        window["timeframeDuration"].update('')
                    else:
                        datas = data.process_file(file_name,timeframeDuration)
                        
                        if file_name==results_list[0]:
                            compiled_data = datas
                            
                        else:
                            compiled_data = pd.concat([compiled_data, datas], axis=0)
        
                            compiled_data = compiled_data.reset_index(drop=True)
                            
                else:  # single frame processing
                    datas = data.process_file_sf(file_name)
        
                    if file_name==results_list[0]:
                        compiled_data = datas
                    else:
                        compiled_data = pd.concat([compiled_data, datas], axis=0)
                        compiled_data = compiled_data.reset_index(drop=True)
    
            
            if compiled_data is not None:
                if "Time" not in compiled_data.columns:  # single frame plotting
                    plots = plot.plot_data_sf(compiled_data,window=window)
                    data.get_save_folder(data=compiled_data, plot=plots)
                else:  # movies plotting
                    plots = plot.plot_data(compiled_data,window=window)
                    data.get_save_folder(data=compiled_data, plot=plots)
                    z = 0
            else:
                sg.popup_error("No valid data to process.")
            file_list = []
            compiled_data = None
            
        if event == "Plot":
            final_df, plotcompiled = plot.plot_data_compiled(values=values,window=window)

        if event == "Save plot to":
            if plotcompiled:
                plot.save_compiled_plot(data=final_df, plot=plotcompiled,stat_export=stat_export)
            else:
                sg.popup_error('Plot before saving maybe?')
        # iterator for number of cells in variable replacement

        if event == "Normalize and plot":

            final_df, plotcompiled = plot.plot_data_compiled_norm(values=values,window=window)
        
        if event == "Run test":
            
            imagesPath=values["ImageTest"]
            
            testImage(imagesPath,window,app)

            
            
if __name__ == "__main__":
    main()
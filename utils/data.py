import os
import re
import numpy as np
import pandas as pd
from utils import data
import pyclesperanto_prototype as cle
import tifffile
import shutil
import json 
import PySimpleGUI as sg

def createTestJson(testParams,path):
    ij_path,params,gpu=readParameters()
    data = {
    "ij_path": ij_path,
    "R_HOME": "/usr/lib/R/",
    "R_LIBS": "/home/adminelson/R/x86_64-pc-linux-gnu-library/4.3/",
    "parameters": {
        "median_radius": testParams["median_radius"],
        "max_filter_size": testParams["max_filter_size"],
        "top_hat_radius": testParams["top_hat_radius"],
        "closing_radius1": testParams["closing_radius1"],
        "closing_radius2": testParams["closing_radius2"],
        "dilation_radius1": testParams["dilation_radius1"],
        "dilation_radius2": testParams["dilation_radius2"],
        "erosion_radius": testParams["erosion_radius"],
        "thresholdtype":testParams["thresholdtype"],
        "vmin": testParams["vmin"],
        "vmedian": testParams["vmedian"],
        "biomedian": testParams["biomedian"],
        "biotophat": testParams["biotophat"],
        "dontprocess": testParams["dontprocess"],
    },
    "selected_gpu": gpu
    }
    
    # Specify the file path where you want to save the JSON
    file_path = path
    
    # Save JSON to file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    
def readParameters():
    with open('config.json', 'r') as file:
        config = json.load(file)
    #Image-J interface
    ij_path = config["ij_path"]

    # R Interface Libraries
    os.environ['R_HOME'] = config["R_HOME"]
    os.environ["R_LIBS"] = config["R_LIBS"]

    #option parameters
    params=config["parameters"]
    gpu=config["selected_gpu"]
    return ij_path,params,gpu


def jsonProof(config,savepath): 
    # Define the source and destination file paths
    source_file = config
    destination_file = savepath
    # Copy the file
    shutil.copy(source_file, destination_file)
    
def singleFrameDataMeasures(image_biosensor,membranes,intracellular,img_pi_path,img_biosensor_path,day):
    stat_mb = cle.statistics_of_labelled_pixels(
        image_biosensor, membranes)
    stat_cyt = cle.statistics_of_labelled_pixels(
        image_biosensor, intracellular)
    table_mb_temp = pd.DataFrame(
        stat_mb)[['label', 'mean_intensity']]
    basename_temp = os.path.basename(img_pi_path)
    basename_temp = basename_temp.replace(basename_temp[0:3], "")
    table_mb_temp = pd.concat(
        [table_mb_temp, pd.DataFrame({'Files': [basename_temp] * 3})], axis=1)
    table_mb_temp = table_mb_temp.rename(
        columns={'mean_intensity': "mean_intensity_membranes"})
    table_cyt_temp = pd.DataFrame(
        stat_cyt)[['label', 'mean_intensity']]
    table_cyt_temp = table_cyt_temp.rename(
        columns={'mean_intensity': "mean_intensity_intracellular"})
    # do not compile table, save for each indivual binome for use in compile for conditions!
    # add ratio and compile
    ratios = table_mb_temp["mean_intensity_membranes"] / \
        table_cyt_temp["mean_intensity_intracellular"]
    table_ratio = pd.concat([table_cyt_temp, pd.DataFrame(table_mb_temp["mean_intensity_membranes"]), pd.DataFrame(
        {"Ratio mb/intra": ratios}), pd.DataFrame(table_mb_temp["Files"])], axis=1)
    # create output folder if not created
    path = os.path.dirname(img_biosensor_path)
    path2 = os.path.dirname(path)
    if not os.path.exists(path2+"/output_single_frame_analysis_"+day):
        os.mkdir(path2+"/output_single_frame_analysis_"+day)

    table_ratio.to_csv(
        f'{path2}/output_single_frame_analysis_{day}/{basename_temp}_analysis.csv')
    
    #export segmentations
    path_mb=f'{path2}/output_single_frame_analysis_{day}/{basename_temp}_membranes-segmentation.tif'
    tifffile.imwrite(path_mb,cle.pull(membranes))
    path_mb=f'{path2}/output_single_frame_analysis_{day}/{basename_temp}_intracellular-segmentation.tif'
    tifffile.imwrite(path_mb,cle.pull(intracellular))
    path_json=f'{path2}/output_single_frame_analysis_{day}/LRC_parameters.json'
    data.jsonProof(config='config.json',savepath=path_json)
    

def adjustTimeTracer(dataframe,folder_path,version,erosionfactor,date_str,savename):
    print("adjusttracerenter::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    # correct frame negative/positive or control/treatment according to parameters file
    # open file
    parameters_file=data.find_file(folder_path, "_parameters.csv")
    if parameters_file is not None:
        img_parameters = pd.read_csv(parameters_file[0])
        print(img_parameters)
        # retrieve significant change value
        change = img_parameters["Value"][4]
        if change == "na" or change == "no significant change found":
            print("user did not required tracer or no significant change was found")
        else:
            # modify frame columns
            dataframe.index = dataframe.index-change
            dataframe.to_csv(folder_path+'/results-LRC'+str(version) +
                            '-erode_'+str(erosionfactor)+"-"+date_str+savename+'.csv')
            # modify frame columns


def folderToAnalyze(values):
    nb_folders_toanalyse = 0
    for folder_nb in np.arange(1, 5, 1):
        folder_path = values["-FOLDER"+str(folder_nb)+"-"]
        if len(folder_path) > 0:
            nb_folders_toanalyse += 1
    return nb_folders_toanalyse

def filenamesFromPaths(list_files,list_pattern_avoid):
    # extract filenames from full path
    list_filenames = [os.path.basename(path) for path in list_files]
    file_dict = dict(zip(list_filenames, list_files))
    list_fileC2 = [file for file in list(file_dict.keys()) if "C2-" in file]
    list_fileC2 = [item for item in list_fileC2 if all(
        not re.search(pattern, item) for pattern in list_pattern_avoid)]
    list_fileC2 = [file_dict.get(filename) for filename in list_fileC2]
    list_fileC1 = [file for file in list(
        file_dict.keys()) if "C1-" in file]
    list_fileC1 = [item for item in list_fileC1 if all(
        not re.search(pattern, item) for pattern in list_pattern_avoid)]
    list_fileC1 = [file_dict.get(filename) for filename in list_fileC1]
    return list_fileC1,list_fileC2

def compareList(l1, l2):
    """
     Compares two lists to determine if they are equal after sorting.
    
     Parameters:
     l1 (list): The first list to be compared.
     l2 (list): The second list to be compared.
    
     The function performs the following steps:
     1. Sorts both lists in place.
     2. Compares the sorted lists.
    
     Returns:
     str: Returns "Equal" if both lists are identical in terms of elements and order after sorting, 
          otherwise returns "Non equal".
    
     Note:
     - The function modifies the original lists (in-place sorting).
     - Comparison is sensitive to both the elements and their order in the lists.
     """
    l1.sort()
    l2.sort()
    if (l1 == l2):
        return "Equal"
    else:
        return "Non equal"


def dfreplace(df, variable, iterator):
    """
    Replaces unique values in a specified column of a DataFrame with a formatted string and an iterator.
    
    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    variable (str): The column name in the DataFrame where the replacements are to be made.
    iterator (int): An integer used to create a unique replacement string for each unique value in the specified column.
    
    The function performs the following steps:
    1. Identifies unique values in the specified column of the DataFrame.
    2. Creates a list of replacement strings, each formatted as "cell-{iterator+1}", incrementing the iterator for each unique value.
    3. Replaces each unique value in the specified column with the corresponding formatted string.
    4. Performs the replacement in place.
    
    Returns:
    tuple: A tuple containing two elements:
        - df (DataFrame): The modified DataFrame with replaced values.
        - iterator (int): The updated iterator after all replacements.
    
    Note:
    - The function modifies the original DataFrame.
    - The iterator is incremented for each unique value in the specified column.
    """
    listtoreplace = list(np.unique(df[variable]))
    listreplacement = []
    for replace in listtoreplace:
        listreplacement.append(f"cell-{iterator+1}")
        iterator += 1
    df.replace(listtoreplace, listreplacement, inplace=True)
    return df, iterator


def df_excel_friendly(df):
    """
    Transforms a DataFrame into a format suitable for Excel display by pivoting it.
    
    Parameters:
    df (DataFrame): The pandas DataFrame to be transformed.
    
    The function performs the following steps:
    1. Pivots the DataFrame based on the 'Time' column as the index and the 'File' column as the columns.
    2. The values in the pivoted DataFrame are filled with the values from the 'Average' column.
    
    Returns:
    DataFrame: A pivoted DataFrame with 'Time' as the index, unique values from 'File' as columns, 
               and 'Average' values as the data in the table.
    
    Note:
    - The function assumes the existence of 'Time', 'File', and 'Average' columns in the input DataFrame.
    - The original DataFrame is not modified; a new pivoted DataFrame is returned.
    """
    # df["Cell-File"] = df['Average'].astype(str) +"-"+ df["File"]
    df = df.pivot(index='Time', columns="File")['Average']
    return df

def update_console(window, console, message):
    """
    Updates a console-like GUI element with a new message.

    Parameters:
    window (sg.Window): The PySimpleGUI window object containing the console element.
    console (str): The key for the console element within the window.
    message (str): The message to be appended to the console.

    The function appends the provided message to the specified console element in the PySimpleGUI window.
    """
    window[console].print(message)  # Append message to the Multiline element
    
def get_save_folder(data, plot):
    """
    Opens a GUI window for selecting a folder to save a DataFrame and its associated plot.

    Parameters:
    data (DataFrame): The DataFrame to be saved.
    plot (matplotlib.figure.Figure): The plot object to be saved.

    The function performs the following steps:
    1. Creates a GUI layout for folder selection.
    2. Opens a window with the defined layout.
    3. Waits for user interaction in the GUI.
    4. If 'Select' is clicked, saves the plot as an SVG file and the data as a CSV file in the chosen folder.
    5. Closes the window after saving the files or if 'Cancel' is clicked.

    Returns:
    None: The function does not return a value but saves files based on user input.

    Note:
    - The function uses PySimpleGUI for the GUI components.
    - The plot is saved in SVG format, and the data is saved in CSV format.
    """
    layout = [[sg.Text("Select folder to save the compiled table and plot:")],
              [sg.InputText(key="Folder"), sg.FileSaveAs('Save As')],
              [sg.Button("Select"), sg.Button("Cancel")]]

    window3 = sg.Window('Saving plot and dataframe', layout)

    while True:
        event3, values3 = window3.read()
        if event3 in (sg.WINDOW_CLOSED, "Cancel"):
            window3.close()
            return None

        if event3 == "Select":
            foldy = values3.get('Folder')
            if foldy:
                plot.savefig(foldy+"_plot.svg")
                data.to_csv(foldy+'_results-compile.csv',index=False)
            window3.close()

    return None


def search_pattern_recursive(data, pattern):
    """
    Recursively searches for a pattern in a nested list structure.

    Parameters:
    data (list): The nested list structure to search through.
    pattern (str): The regex pattern to search for.

    The function performs the following steps:
    1. Iterates through each item in the list.
    2. If an item is a list, recursively searches within that list.
    3. If an item is a string, checks for a match with the pattern.
    4. Collects all matching items.

    Returns:
    list: A list of all items that match the pattern.

    Note:
    - The function uses regular expressions to match patterns.
    """
    matches = []

    for item in data:
        if isinstance(item, list):  # If the item is a list, recurse into it
            matches.extend(search_pattern_recursive(item, pattern))
        # If the item is a string and matches the pattern
        elif isinstance(item, str) and re.search(pattern, item):
            matches.append(item)

    return matches


def process_file(file_name,timeframeDuration):
    """
    Processes a CSV file to extract and average data from columns matching a specific pattern.

    Parameters:
    file_name (str): The path to the CSV file to be processed.

    The function performs the following steps:
    1. Reads the CSV file into a DataFrame.
    2. Searches for columns matching the pattern "ratio_mb/cyt".
    3. Averages the data across these columns.
    4. Creates a time column based on the 'Frames' column, scaled by a factor timeframeDuration.
    5. Combines the time and averaged data into a new DataFrame.
    6. Renames columns appropriately and adds a 'File' column with the file's base name.

    Returns:
    DataFrame: A DataFrame containing the time and averaged data, along with the file name.

    Note:
    - The function assumes the presence of a 'Frames' column in the CSV file for time data.
    """
    all_data = None   # Initialization
    df = pd.read_csv(file_name)
    time_col = df["Frames"]*timeframeDuration
    df = df.rename(columns={'Frames': 'Frame'})
    all_data = pd.concat([time_col, df], axis=1)
    all_data = all_data.rename(columns={'Frames': 'Time'})
    all_data['File'] = os.path.basename(os.path.dirname(file_name))
    return all_data


def process_file_sf(file_name):
    """
    Processes a CSV file to calculate the average of a specific column.

    Parameters:
    file_name (str): The path to the CSV file to be processed.

    The function performs the following steps:
    1. Reads the CSV file into a DataFrame.
    2. Calculates the mean of the 'Ratio mb/intra' column.
    3. Creates a new DataFrame with the file name and the calculated average.

    Returns:
    DataFrame: A DataFrame containing the file name and the average of the 'Ratio mb/intra' column.

    Note:
    - The function assumes the presence of a 'Ratio mb/intra' column in the CSV file.
    """
    df = pd.read_csv(file_name)
    average = df['Ratio mb/intra'].mean()
    all_data = pd.DataFrame(
        {"File": [os.path.basename(file_name)], "Average ratios": [average]})
    return all_data


def find_file_recursive(folder, pattern):
    """
    Recursively searches for files in a directory matching a specific pattern.

    Parameters:
    folder (str): The directory to search in.
    pattern (str): The file pattern to match.

    The function performs the following steps:
    1. Traverses the directory structure starting from 'folder'.
    2. Collects all files that end with the specified 'pattern'.

    Returns:
    list: A list of paths to files that match the pattern.

    Note:
    - The function uses os.walk to traverse directories.
    """
    files = []

    # os.walk yields a 3-tuple (dirpath, dirnames, filenames) for each directory it visits
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(pattern):
                # Use os.path.join to create full path to file
                full_path = os.path.join(dirpath, filename)
                files.append(full_path)
    return files
# Get the shape of the data, the coordinate pairs are (start index, size)


def find_file(folder, pattern):
    """
    Searches for files in a directory matching a specific pattern.

    Parameters:
    folder (str): The directory to search in.
    pattern (str): The file pattern to match.

    The function performs the following steps:
    1. Lists all files in the specified 'folder'.
    2. Filters and returns files that contain the 'pattern'.

    Returns:
    list: A list of paths to files that match the pattern.

    Note:
    - This function does not search subdirectories.
    """
    list_files = os.listdir(folder)
    i = 0
    for file in list_files:
        list_files[i] = os.path.join(folder, file)
        i += 1
    files = [file for file in list_files if pattern in file]
    return files


def exp_decreasing(x, a, b, c):
    """
    Defines an exponentially decreasing function.

    Parameters:
    x (numeric): The independent variable.
    a, b, c (numeric): Parameters of the exponential function.

    Returns:
    numeric: The value of the exponential function a * exp(-b * x) + c at x.

    Note:
    - This function is typically used in curve fitting or modeling scenarios.
    """
    return a * np.exp(-b * x) + c


def movieRatios(table_cyt,table_mb,window,values):

    for label in np.unique(table_cyt["label"]):
        if int(label) == 1:
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
            df_ratio_av=df_ratio.copy()
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
            
            df_ratio = pd.concat([df_ratio, df_ratio2], axis=0)##at the end of each other by col
    return df_ratio
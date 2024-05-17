import PySimpleGUI as sg
import pyclesperanto_prototype as cle
def choose_gpu_popup():
    # List available GPUs
    available_gpus = cle.available_device_names()

    # Define layout for the popup
    layout = [
        [sg.Text("Choose a GPU")],
        [sg.Listbox(values=available_gpus, size=(30, len(available_gpus)), key='gpu', enable_events=True)],
        [sg.Button("Select"), sg.Button("Cancel")]
    ]

    # Create the popup window
    gpu_pop= sg.Window("GPU Selection", layout)

    # Event loop for the popup
    while True:
        event, values = gpu_pop.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            break
        elif event == "Select":
            chosen_gpu = values["gpu"][0] if values["gpu"] else None
            gpu_pop.close()
            return chosen_gpu

    gpu_pop.close()
    return None
import os
import json
import sys
from utils import data
def open_parameter_popup():
    
    with open(f'{os.getcwd()}/config.json', 'r') as file:
        config = json.load(file)
    #option parameters
    params=config["parameters"]
    popup_layout = [
        [sg.Text('Enter the values:')],
        [sg.Text('Median Radius'), sg.Input(params["median_radius"], key='median_radius')],
        [sg.Text('Max Filter Size'), sg.Input(params['max_filter_size'], key='max_filter_size')],
        [sg.Text('Top Hat Radius'), sg.Input(params['top_hat_radius'], key='top_hat_radius')],
        [sg.Text('Closing Radius 1'), sg.Input(params["closing_radius1"], key='closing_radius1')],
        [sg.Text('Closing Radius 2'), sg.Input(params["closing_radius2"], key='closing_radius2')],
        [sg.Text('Dilation Radius 1'), sg.Input(params['dilation_radius1'], key='dilation_radius1')],
        [sg.Text('Dilation Radius 2'), sg.Input(params['dilation_radius2'], key='dilation_radius2')],
        [sg.Text('Erosion Radius'), sg.Input(params["erosion_radius"], key='erosion_radius')],
        [sg.Text('Threshold for vacuoles and background'), sg.Combo(["Otsu Threshold", "Median Threshold"], default_value=params["thresholdtype"], key='thresholdtype')],
        [sg.Text('VMin'), sg.Input(params["vmin"], key='vmin')],
        [sg.Text('VMedian'), sg.Input(params["vmedian"], key='vmedian')],
        [sg.Text('BioMedian'), sg.Input(params["biomedian"], key='biomedian')],
        [sg.Text('BioTopHat'), sg.Input(params["biotophat"], key='biotophat'),
         sg.Checkbox("No intensity filter", default=params["dontprocess"], key='dontprocess')],
        [sg.Button("Select GPU", key='Select GPU'),sg.Button("Default")],
        [sg.Input(key="loadparams"),sg.FilesBrowse("Import Parameters"),sg.Button("Load")],
        [sg.Submit(), sg.Cancel()]
    ]

    selected_gpu = None

    popup_window = sg.Window('Enter Parameters', popup_layout)

    while True:
        event, values = popup_window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            popup_window.close()
            return None
        elif event == 'Select GPU':
            new_gpu_selection = choose_gpu_popup()
            if new_gpu_selection is not None:
                selected_gpu = new_gpu_selection
                print("Selected GPU:", selected_gpu)
                
        elif event == 'Submit':
            if values is not None:
                # Convert numeric values to integers
                for key, value in values.items():
                    if isinstance(value, str):
                        try:
                            # Attempt to convert to integer
                            values[key] = int(value)
                        except ValueError:
                            # Handle non-integer values
                            pass
                if selected_gpu is not None:
                    data_to_save = {
                        "parameters": values,
                        "selected_gpu": selected_gpu
                    }
                    
                else:
                    data_to_save = {
                        "parameters": values
                    }
                print(values)
                print("Calling append_to_json_file")
                # Append data to JSON file here
                file_name = 'config.json'
                try:
                    with open(file_name, 'r') as file:
                        datas = json.load(file)
                except FileNotFoundError:
                    print(f"{file_name} not found. Creating a new one.")
                    datas = {}
                except json.JSONDecodeError:
                    print(f"Error reading {file_name}. Starting with empty data.")
                    datas = {}

                datas.update(data_to_save)
                try:
                    with open(file_name, 'w') as file:
                        json.dump(datas, file, indent=4)
                    print(f"Data written to {file_name}")
                except IOError as e:
                    print(f"Error writing to {file_name}: {e}")
            break
        #Implement a loop an maybe number stuff
        elif event == "Load":
            fileimport=str(values["loadparams"])
            file_name = 'config.json'
            # Open the JSON file
            with open(fileimport, 'r') as f:
                datacustom = json.load(f)
            # Save the modified data 
            with open(file_name, 'w') as f:
                json.dump(datacustom, f, indent=4)
            with open(f'{os.getcwd()}/config.json', 'r') as file:
                config = json.load(file)
            params=config["parameters"]
            # Update the values in the window from the loaded JSON data
            for param in list(params.keys()):
                popup_window[param].update(params[param])
            popup_window.refresh()
            
        elif event == "Default":
            fileimport="config_default.json"
            file_name = 'config.json'
            # Open the JSON file
            with open(fileimport, 'r') as f:
                datacustom = json.load(f)
            # Save the modified data 
            with open(file_name, 'w') as f:
                json.dump(datacustom, f, indent=4)
            with open(f'{os.getcwd()}/config.json', 'r') as file:
                config = json.load(file)
            params=config["parameters"]
            # Update the values in the window from the loaded JSON data
            for param in list(params.keys()):
                popup_window[param].update(params[param])
            popup_window.refresh()


    popup_window.close()

def launch_main_gui(name):
    ORANGE = '#e97d62'
    WHITE = '#e5decf'
    GREY = '#cfcfcf'
    DARK_GREY = '#404040'
    sg.LOOK_AND_FEEL_TABLE['MyCreatedTheme'] = {'BACKGROUND': GREY, 
                                            'TEXT': DARK_GREY, 
                                            'INPUT': WHITE, 
                                            'TEXT_INPUT': DARK_GREY, 
                                            'SCROLL': '#b3c1cd', 
                                            'BUTTON': (WHITE, ORANGE), 
                                            'PROGRESS': (GREY, ORANGE), 
                                            'BORDER': 1, 'SLIDER_DEPTH': 0,'PROGRESS_DEPTH': 0, } 
      
    # Switch to use your newly created theme 
    sg.theme('MyCreatedTheme') 
    layout0 = [[sg.Text("Select folder containing czi files to split (it's recursive):")],
               [sg.Input(key="-FOLDER0-"), sg.FolderBrowse(), sg.Button("Split")],
               [sg.Text("Input a pattern, otherwise all the .czi files in the specified folder will be splitted",size=(65,1)),
                sg.Input(key="patternsplit")],
               [sg.Text("Select folder containing splitted channels (it's recursive):")],
               [sg.Input(key="-FOLDER00-"), sg.FolderBrowse(),
                sg.Button("3D-Registration")],
               [sg.Text(
                   "Select folder containing splitted and stabilized channels (it's recursive):")],
               [sg.Input(key="-FOLDER0000-"), sg.FolderBrowse(),
                sg.Button("Manual ROI selection"),sg.Checkbox("Analyse background tracer",key="analysetracer")],
               [sg.Multiline(size=(120, 20), key='-CONSOLE0-',
                             autoscroll=True, disabled=True)]
               ]
    
    layout1 = [
        [sg.Column([
            [sg.Text(
                "Select a folder containing stabilized ROI, pi.tif and fluo.tif:", size=(65, 1))],
            [sg.Input(key="-FOLDER1-"), sg.FolderBrowse()],
            [sg.Text("Select a folder containing stabilized ROI, pi.tif and fluo.tif:")],
            [sg.Input(key="-FOLDER2-"), sg.FolderBrowse()],
            [sg.Text("Select a folder containing stabilized ROI, pi.tif and fluo.tif:")],
            [sg.Input(key="-FOLDER3-"), sg.FolderBrowse()],
            [sg.Text("Select a folder containing stabilized ROI, pi.tif and fluo.tif:")],
            [sg.Input(key="-FOLDER4-"), sg.FolderBrowse()],
            [sg.Text("Select the line you are analysing"), sg.Combo(
                ["991", "604", "564"], default_value='991', readonly=True, key="erosion")],
            [sg.Text("Do you want to save the stitched image?"),
             sg.Checkbox('', default=True, key='save_stitched')],
            [sg.Multiline(size=(50, 10), key='-CONSOLE-',
                          autoscroll=True, disabled=True)],
            [sg.Button("Run image processing"), sg.Button(
                "Clear"), sg.Button("Cancel"),
                sg.Button("Options", key='Options1')]
        ]), sg.Canvas(key='-CANVAS-')]
    ]
    
    layout12 = [
        [sg.Column([
            [sg.Text("Select a folder containing your czi files:", size=(65, 1))],
            [sg.Input(key="-FOLDER1122-"), sg.FolderBrowse(),sg.Button("Split Channels")],
            [sg.Text("Input a pattern, otherwise all the .czi files in the specified folder will be splitted",size=(65,1)),
             sg.Input(key="patternsplit")],
            [sg.Text("Select a folder containing your splitted channel files:", size=(65, 1))],
            [sg.Input(key="-FOLDER12-"), sg.FolderBrowse()],
            [sg.Text("Select the line you are analysing"), sg.Combo(
                ["991", "604", "564"], default_value='991', readonly=True, key="erosion")],
            [sg.Multiline(size=(50, 10), key='-CONSOLE1-',
                          autoscroll=True, disabled=True)],
            [sg.Button("Run image processing Single frame"),
             sg.Button("Clear"), sg.Button("Cancel"),
             sg.Button("Options", key='Options2')]
        ]), sg.Canvas(key='-CANVAS1-')]
    ]
    
    layout2 = [[sg.Text("Select folder containing results-LRC files:")],
               [sg.InputText(key="File"), sg.FolderBrowse()],
               [sg.Text("Indicate you timeframe interval in minutes (leave default if singleframes)"),
               sg.InputText("2",key='timeframeDuration', enable_events=True,size=(4,1))],
               [sg.Button("Compile")],
               [sg.Canvas(key='-CANVAS2-', size=(400, 400))]]
    
    layout3 = [[sg.Text("Select compiled data in the csv format:", size=(56, 1)), sg.Text("Enter the name of your conditions")],
               [sg.InputText(key="PlotFile1"), sg.FileBrowse(file_types=(
                   ("CSV Files", "*.csv"),)), sg.InputText(key="Condition1")],
               [sg.InputText(key="PlotFile2"), sg.FileBrowse(file_types=(
                   ("CSV Files", "*.csv"),)), sg.InputText(key="Condition2")],
               [sg.InputText(key="PlotFile3"), sg.FileBrowse(file_types=(
                   ("CSV Files", "*.csv"),)), sg.InputText(key="Condition3")],
               [sg.InputText(key="PlotFile4"), sg.FileBrowse(file_types=(
                   ("CSV Files", "*.csv"),)), sg.InputText(key="Condition4")],
               [sg.Canvas(key='-CANVAS3-'), sg.Canvas(key='-CANVAS4-')],
               [sg.Checkbox('Non parametric statistics using R nparcomp functions (single frame only)', default=False,key="statsbox")],
               [sg.Button("Plot"), sg.Button("Save plot to"), sg.Button("Normalize and plot")]]
    
    layoutTest = [[sg.Canvas(key='-CANVAS5-')],#Plot window
                  [sg.InputText(key="ImageTest"), sg.FileBrowse(file_types=(
                      ("tif Files", "*.tif"),))],#Import file path

                  [sg.Button("Run test")]
                  
                  
        ]
    
    # Create tabs
    tab0 = sg.Tab("Image preprocessing", layout0)
    tab1 = sg.Tab("Image analysis", layout1)
    tab12 = sg.Tab("Image analysis single frames", layout12)
    tab2 = sg.Tab("Data compilation", layout2)
    tab3 = sg.Tab("Plotting", layout3)
    tab4= sg.Tab("Test filters",layoutTest)
    
    # Combine tabs into a TabGroup
    layout = [
        [sg.TabGroup([[tab0, tab1, tab12, tab2, tab3, tab4]])],
    ]
    
    window = sg.Window(name, icon="icon.png", layout=layout)
    return window
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
from utils import data
def open_parameter_popup():
    
    with open(f'{os.getcwd()}/config.json', 'r') as file:
        config = json.load(file)
    #option parameters
    params1=config["parameters"]
    params=data.process_parameters(data=params1)
    popup_layout = [
        [sg.Text('Enter the values:')],
        [sg.Text('Median Radius'), sg.InputText(str(params[0]), key='median_radius')],
        [sg.Text('Max Filter Size'), sg.InputText(str(params[1]), key='max_filter_size')],
        [sg.Text('Top Hat Radius'), sg.InputText(str(params[2]), key='top_hat_radius')],
        [sg.Text('Closing Radius 1'), sg.InputText(str(params[3]), key='closing_radius1')],
        [sg.Text('Closing Radius 2'), sg.InputText(str(params[4]), key='closing_radius2')],
        [sg.Text('Dilation Radius 1'), sg.InputText(str(params[5]), key='dilation_radius1')],
        [sg.Text('Dilation Radius 2'), sg.InputText(str(params[6]), key='dilation_radius2')],
        [sg.Text('Erosion Radius'), sg.InputText(str(params[7]), key='erosion_radius')],
        [sg.Text('VMin'), sg.InputText(str(params[8]), key='vmin')],
        [sg.Text('VMedian'), sg.InputText(str(params[9]), key='vmedian')],
        [sg.Text('BioMedian'), sg.InputText(str(params[10]), key='biomedian')],
        [sg.Text('BioTopHat'), sg.InputText(str(params[11]), key='biotophat')],
        [sg.Button("Select GPU", key='Select GPU')],
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
               [sg.Text("Select folder containing splitted channels (it's recursive):")],
               [sg.Input(key="-FOLDER00-"), sg.FolderBrowse(),
                sg.Button("3D-Registration")],
               [sg.Text(
                   "Select folder containing splitted and stabilized channels (it's recursive):")],
               [sg.Input(key="-FOLDER000-"), sg.FolderBrowse(),
                sg.Button("Tracer-analysis")],
               [sg.Text(
                   "Select folder containing splitted and stabilized channels (it's recursive):")],
               [sg.Input(key="-FOLDER0000-"), sg.FolderBrowse(),
                sg.Button("Manual ROI selection")],
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
            [sg.Text("Select a folder containing your biosensor images:", size=(65, 1))],
            [sg.Input(key="-FOLDER12-"), sg.FolderBrowse()],
            [sg.Text("Select a folder containing your propidium iodide images:")],
            [sg.Input(key="-FOLDER22-"), sg.FolderBrowse()],
            [sg.Text("Select the line you are analysing"), sg.Combo(
                ["991", "604", "564"], default_value='991', readonly=True, key="erosion")],
            [sg.Multiline(size=(50, 10), key='-CONSOLE1-',
                          autoscroll=True, disabled=True)],
            [sg.Button("Run image processing Single frame"),
             sg.Button("Clear"), sg.Button("Cancel"),
             sg.Button("Options", key='Options2')]
        ]), sg.Canvas(key='-CANVAS1-')]
    ]
    
    layout2 = [[sg.Text("Select CSV files:")],
               [sg.InputText(key="File"), sg.FileBrowse(
                   file_types=(("CSV Files", "*.csv"),))],
               [sg.Button("Add"), sg.Button("Clear list"), sg.Button("Compile")],
               [sg.Multiline(size=(150, 10), key='-CONSOLE2-',
                             autoscroll=True, disabled=True)],
               [sg.Canvas(key='-CANVAS2-')]]
    
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
    
    # Create tabs
    tab0 = sg.Tab("Image preprocessing", layout0)
    tab1 = sg.Tab("Image analysis", layout1)
    tab12 = sg.Tab("Image analysis single frames", layout12)
    tab2 = sg.Tab("Data compilation", layout2)
    tab3 = sg.Tab("Plotting", layout3)
    
    # Combine tabs into a TabGroup
    layout = [
        [sg.TabGroup([[tab0, tab1, tab12, tab2, tab3]])],
    ]
    
    window = sg.Window(name, icon="icon.png", layout=layout)
    return window
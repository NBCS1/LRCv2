import seaborn as sns
from datetime import date

import pandas as pd
import json
import os
with open('config.json', 'r') as file:
    config = json.load(file)
#Image-J interface
ij_path = config["ij_path"]

# R Interface Libraries
os.environ['R_HOME'] = config["R_HOME"]
os.environ["R_LIBS"] = config["R_LIBS"]
import rpy2.rinterface as rinterface
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, Formula
def plot_nparcomp(file_path, column_cond, column_values, column_replicate, compare,statsbox,listfiles):
    """
    Conducts non-parametric statistical comparisons and visualizes the results using boxplots and swarmplots.
    
    Parameters:
    file_path (str or DataFrame): The path to the data file or a pandas DataFrame containing the data.
    column_cond (int): The index of the column in the DataFrame that contains the condition/group labels.
    column_values (int): The index of the column in the DataFrame that contains the values to be analyzed.
    column_replicate (int): The index of the column in the DataFrame that contains replicate identifiers.
    compare (str): A string specifying the type of comparison ('all' for all pairwise comparisons or 
                   another value for specific comparisons, e.g., 'Dunnett').
    
    The function performs the following steps:
    1. Converts the pandas DataFrame to an R DataFrame.
    2. Loads necessary R packages for non-parametric comparison.
    3. Sets up and runs the nparcomp test in R, handling different numbers of conditions.
    4. Extracts p-values and performs compact letter display calculations if applicable.
    5. Visualizes the results using seaborn's boxplot and swarmplot.
    6. Adds statistical significance annotations to the plots.
    
    Returns:
    matplotlib.figure.Figure: A matplotlib figure object containing the generated plot.
    """
    if statsbox and len(listfiles)>1:
        global stat_export
        global d1
        global test_result
        global cld_result
        d1 = None
        test_result = None
        cld_result = None
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        df = file_path
        # Convert pandas dataframe to R dataframe
        with localconverter(ro.default_converter + pandas2ri.converter):
            rdf = ro.conversion.py2rpy(df)
    
        # Load the necessary R packages
        nparcomp = ro.packages.importr('nparcomp')
        rcomp = ro.packages.importr('rcompanion')
    
        # Set up the formula for the nparcomp test
        formula = Formula("y~x")
    
        # Run the nparcomp test on the data
        env = formula.environment
        env['x'] = rdf[column_cond]
        env['y'] = rdf[column_values]
    
        ttest = ro.r['npar.t.test']
        conditions_nb = len(set(df.iloc[:, column_cond]))
        if conditions_nb > 2:
            if compare == 'all':
                test_result = nparcomp.nparcomp(formula, rdf, **{"type": "Tukey"})
                # Extract the p-values from the test result object
                p_values = test_result.rx2('Analysis')[5]
                cond = test_result.rx2('connames')
            else:
                test_result = nparcomp.nparcomp(
                    formula, rdf, **{"type": "Dunnett"})
                # Extract the p-values from the test result object
                p_values = test_result.rx2('Analysis')[5]
                cond = test_result.rx2('connames')
    
        else:
            test_result = ttest(formula, rdf, **{'method': 'permu'})
            # Extract the p-values from the test result object
            p_values = test_result.rx2('Analysis')[4][0]
    
        if compare == 'all' and conditions_nb > 2:
            with localconverter(ro.default_converter + pandas2ri.converter):
                cond_P = ro.conversion.rpy2py(cond)
            cond_P = [a.replace('p(', '') for a in cond_P]
            cond_P = [a.replace(')', '') for a in cond_P]
            cond_P = [a.replace(',', '-') for a in cond_P]
            cond_P = pd.DataFrame(cond_P)
            with localconverter(ro.default_converter + pandas2ri.converter):
                cond = ro.conversion.py2rpy(cond_P)
            cond = cond[0]
            # compact letter display calculations
            cld_result = rcomp.cldList(
                rinterface.NULL, rinterface.NULL, cond, p_values)
            print(cld_result)
    
            # Convert the cld result to a pandas dataframe
            with localconverter(ro.default_converter + pandas2ri.converter):
                cld_letter = ro.conversion.rpy2py(cld_result[1])
                cld_groups = ro.conversion.rpy2py(cld_result[0])
    
            # Merge the cld dataframe with the original data dataframe
            merged_df = pd.DataFrame(list(zip(cld_groups, cld_letter)))
    
            # Create a boxplot and swarmplot using seaborn
            # Add the letters from the compact letter display on top of the corresponding boxes
            # retrieve column name!
            means = df.groupby([df.columns[column_cond]])[
                df.columns[column_values]].max()
    
            fig, ax = plt.subplots()
            sns.boxplot(x=df.iloc[:, column_cond],
                        y=df.iloc[:, column_values], data=df, ax=ax)
            sns.swarmplot(x=df.iloc[:, column_cond], y=df.iloc[:, column_values],
                          data=df, hue=df.iloc[:, column_replicate], ax=ax)
            sns.despine(offset=2, trim=True, ax=ax)
            sns.set_style("ticks")
            ax.legend().remove()
            for i in range(len(merged_df)):
                plt.text(i, means.iloc[i]+(0.02/max(df.iloc[:, column_values])),
                         merged_df.iloc[:, 1][i], fontsize=20, ha='center', va='bottom')
    
        else:
            means = df.groupby([df.columns[column_cond]])[
                df.columns[column_values]].max()
    
            fig, ax = plt.subplots()  # Create a figure and an axes
            sns.boxplot(x=df.iloc[:, column_cond],
                        y=df.iloc[:, column_values], data=df, ax=ax)
            sns.swarmplot(x=df.iloc[:, column_cond], y=df.iloc[:, column_values],
                          data=df, hue=df.iloc[:, column_replicate], ax=ax)
            sns.despine(offset=2, trim=True, ax=ax)
            sns.set_style("ticks")
            ax.legend().remove()
            # extract pvalues and transfer in ***
            with localconverter(ro.default_converter + pandas2ri.converter):
                stars = ro.conversion.rpy2py(p_values)
    
            if conditions_nb > 2:
                stars2 = []
                for i in stars:
                    if i < 0.0005:
                        stars2.append('***')
                    elif i < 0.005:
                        stars2.append('**')
                    elif i < 0.05:
                        stars2.append('*')
                    elif i > 0.05:
                        stars2.append('ns')
                for i in range(len(stars)):
                    # put0.1 as a parameter!
                    plt.text(i+1, means.iloc[i+1]+(0.02/max(df.iloc[:, column_values])),
                             stars2[i], fontsize=20, ha='center', va='bottom')
            else:
                if stars < 0.0005:
                    stars2 = '***'
                elif stars < 0.005:
                    stars2 = '**'
                elif stars < 0.05:
                    stars2 = '*'
                elif stars > 0.05:
                    stars2 = 'ns'
                plt.text(1, means.iloc[1]+(0.012/max(df.iloc[:, column_values])),
                         stars2, fontsize=20, ha='center', va='bottom')
    
    
    else:#single frame one compiled file or unticked statbox just plot
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        df = file_path
        means = df.groupby([df.columns[column_cond]])[
            df.columns[column_values]].max()

        fig, ax = plt.subplots()  # Create a figure and an axes
        sns.boxplot(x=df.iloc[:, column_cond],
                    y=df.iloc[:, column_values], data=df, ax=ax)
        sns.swarmplot(x=df.iloc[:, column_cond], y=df.iloc[:, column_values],
                      data=df, hue=df.iloc[:, column_replicate], ax=ax)
        sns.despine(offset=2, trim=True, ax=ax)
        sns.set_style("ticks")
        ax.legend().remove()
        
    plt.close(fig)
    stat_export = True
    return fig  # add object to rebuild stat report

import matplotlib.pyplot as plt
import numpy as np
def singleFrameAnalysisDisplay(membranes, novacuole, intracellular,window):
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # Display images
    ax1.imshow(np.hstack((membranes, novacuole, intracellular)))

    ax1.set_axis_off()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    plt.subplots_adjust(wspace=0, hspace=0,
                        left=0, right=1, bottom=0, top=1)
    plt.show()

    for item in window['-CANVAS1-'].TKCanvas.pack_slaves():
        item.destroy()
    draw_figure(window['-CANVAS1-'].TKCanvas, fig)
    window.refresh()

def testDisplay(membranes, novacuole, intracellular,window):
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # Display images
    ax1.imshow(np.hstack((membranes, novacuole, intracellular)))

    ax1.set_axis_off()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    plt.subplots_adjust(wspace=0, hspace=0,
                        left=0, right=1, bottom=0, top=1)
    plt.show()

    for item in window['-CANVAS5-'].TKCanvas.pack_slaves():
        item.destroy()
    draw_figure(window['-CANVAS5-'].TKCanvas, fig)
    window.refresh()
    return fig

    
def plot_data_compiled(values,window):
    """
    Compiles data from multiple CSV files and generates a plot based on the compiled data.
    
    The function performs the following steps:
    1. Gathers file paths and conditions from a global 'values' dictionary, assuming specific key patterns.
    2. Assigns default condition names if any are unspecified.
    3. Reads data from each file, adds a 'condition' column, and compiles all data into a single DataFrame.
    4. Resets the index of the compiled DataFrame.
    5. Determines the type of plot to generate based on the presence of a 'Time' column in the DataFrame.
    6. For time-series data, creates a line plot with error bars.
    7. For non-time-series data, generates a plot using the 'plot_nparcomp' function.
    8. Updates a GUI window with the generated plot.
    
    Returns:
    tuple: A tuple containing two elements:
        - compiled_df (DataFrame): The compiled DataFrame from all input files.
        - plot (matplotlib.figure.Figure or seaborn.axisgrid.FacetGrid): The generated plot object.
    
    Note:
    - The function relies on a global 'values' dictionary for input file paths and conditions.
    - It assumes specific column names ('Time', 'Average') in the input data.
    - The function is designed to work within a GUI environment, specifically updating elements in a window object.
    """
    compiled_df = None
    listfiles = [f for f in list(values.keys()) if 'PlotFile' in str(f)]
    listpaths = [values[a] for a in listfiles if len(values[a]) > 0]
    listconditionsValues = [f for f in list(
        values.keys()) if 'Condition' in str(f)]
    listconditions = [values[a] for a in listconditionsValues]

    for it in np.arange(0, len(listconditions)):
        if len(listconditions[it]) == 0:
            listconditions[it] = f"condition{it+1}"

    for file, condition in zip(listpaths, listconditions):
        if file == listpaths[0]:
            compiled_df = pd.read_csv(file)
            compiled_df["condition"] = condition
        else:
            temp_df = pd.read_csv(file)
            temp_df['condition'] = condition
            compiled_df = pd.concat([compiled_df, temp_df], axis=0)

    compiled_df = compiled_df.reset_index(drop=True)

    if "Time" in compiled_df.columns:  # movie
        plotcompiled = sns.relplot(
            x="Time", y="Average", kind="line", errorbar='se', data=compiled_df, hue="condition")
        for item in window['-CANVAS3-'].TKCanvas.pack_slaves():
            item.destroy()
        draw_figure(window['-CANVAS3-'].TKCanvas, plotcompiled.fig)
        window.refresh()
        return compiled_df, plotcompiled

    else:  # single frame

        fig = plot_nparcomp(file_path=compiled_df, column_cond=3, column_values=2,
                            column_replicate=1, compare='all',statsbox=values["statsbox"],listfiles=listfiles)  # create object to return

        for item in window['-CANVAS3-'].TKCanvas.pack_slaves():
            item.destroy()
        draw_figure(window['-CANVAS3-'].TKCanvas, fig)
        window.refresh()

        return compiled_df, fig  # export object for full stat report


def plot_data_compiled_norm(values,window):
    """
     Compiles and normalizes data from multiple CSV files and generates plots based on the compiled and normalized data.
    
     The function performs the following steps:
     1. Gathers file paths and conditions from a global 'values' dictionary, assuming specific key patterns.
     2. Assigns default condition names if any are unspecified.
     3. Reads data from each file, adds a 'condition' column, and compiles all data into a single DataFrame.
     4. Resets the index of the compiled DataFrame.
     5. Checks for the presence of a 'Time' column in the DataFrame.
     6. If 'Time' is not present, calls 'plot_data_compiled' function and updates the GUI.
     7. If 'Time' is present, normalizes the data based on control period averages.
     8. Generates two plots: one for the original data and one for the normalized data.
     9. Updates the GUI with both plots.
    
     Returns:
     tuple: A tuple containing two elements:
         - df_merged (DataFrame): The DataFrame with normalized values if 'Time' is present; otherwise, the compiled DataFrame.
         - plotcompiled_norm (matplotlib.figure.Figure or seaborn.axisgrid.FacetGrid): The generated plot object for the normalized data; if 'Time' is not present, the plot object from 'plot_data_compiled'.
    
     Note:
     - The function relies on a global 'values' dictionary for input file paths and conditions.
     - It assumes specific column names ('Time', 'Average') in the input data.
     - The function is designed to work within a GUI environment, specifically updating elements in a window object.
     - Normalization is performed by dividing the 'Average' values by the control period average for each condition.
     """
    compiled_df = None
    listfiles = [f for f in list(values.keys()) if 'PlotFile' in str(f)]
    
    listpaths = [values[a] for a in listfiles if len(values[a]) > 0]
    listconditionsValues = [f for f in list(
        values.keys()) if 'Condition' in str(f)]
    listconditions = [values[a] for a in listconditionsValues]

    for it in np.arange(0, len(listconditions)):
        if len(listconditions[it]) == 0:
            listconditions[it] = f"condition{it+1}"
    
    for file, condition in zip(listpaths, listconditions):
        if file == listpaths[0]:
            compiled_df = pd.read_csv(file)
            compiled_df["condition"] = condition
        else:
            temp_df = pd.read_csv(file)
            temp_df['condition'] = condition
            compiled_df = pd.concat([compiled_df, temp_df], axis=0)
    compiled_df = compiled_df.reset_index(drop=True)
    
    if "Time" not in compiled_df.columns:
        final_df, plotcompiled = plot_data_compiled()
        for item in window['-CANVAS3-'].TKCanvas.pack_slaves():
            item.destroy()
        draw_figure(window['-CANVAS3-'].TKCanvas, plotcompiled)
        window.refresh()
        return final_df, plotcompiled
    else:
        #read trace significant value
        # Filter the DataFrame to only include the control period (Time from 0 to 4)

        control_df = compiled_df[compiled_df['Time'].between(
            compiled_df['Time'][0], -2)]

        # Calculate the average value for each cell during the control period
        control_avg = control_df.groupby(['condition'])[
            'Average'].mean().reset_index()

        # Merge the control average back into the original DataFrame
        df_merged = pd.merge(compiled_df, control_avg, on=[
                             'condition'], suffixes=('', '_control'))

        # Perform the normalization: value / average_control_value
        df_merged['value_normalized'] = df_merged['Average'] / \
            df_merged['Average_control']

        plotcompiled = sns.relplot(
            x="Time", y="Average", kind="line", errorbar='sd', data=compiled_df, hue="condition")
        plotcompiled_norm = sns.relplot(
            x="Time", y="value_normalized", kind="line", errorbar='sd', data=df_merged, hue="condition")
        # Add a red dashed line at x=0 to each Axes in the FacetGrid
        for ax in plotcompiled.axes.flat:
            ax.axvline(0, color='r', linestyle='--')

        for ax in plotcompiled_norm.axes.flat:
            ax.axvline(0, color='r', linestyle='--')

        for item in window['-CANVAS3-'].TKCanvas.pack_slaves():
            item.destroy()
        draw_figure(window['-CANVAS3-'].TKCanvas, plotcompiled.fig)
        window.refresh()
        for item in window['-CANVAS4-'].TKCanvas.pack_slaves():
            item.destroy()
        draw_figure(window['-CANVAS4-'].TKCanvas, plotcompiled_norm.fig)
        window.refresh()
        return df_merged, plotcompiled_norm

import PySimpleGUI as sg
def save_compiled_plot(data, plot,stat_export):
    """
    Opens a GUI window for saving a plot and its associated data to files.
    
    Parameters:
    data (DataFrame): The data associated with the plot.
    plot (matplotlib.figure.Figure): The plot object to be saved.
    
    The function performs the following steps:
    1. Creates a GUI layout for saving the plot, with options to specify the file path and type.
    2. Opens a window with the defined layout.
    3. Waits for user interaction in the GUI.
    4. If 'Save' is clicked, saves the plot as an SVG file and the data as a CSV file at the specified location.
    5. If 'stat_export' is True, also saves a full statistical report as a text file.
    6. Closes the window after saving the files or if 'Cancel' is clicked.
    
    Returns:
    None: The function does not return a value but saves files based on user input.
    
    Note:
    - The function relies on global variables 'stat_export', 'd1', 'test_result', and 'cld_result' for saving the statistical report.
    - It uses PySimpleGUI for the GUI components.
    - The function handles file path input and provides error messages for invalid inputs.
    - The plot is saved in SVG format, and the data is saved in CSV format.
    """
    layout = [
        [sg.Text("Select folder to save the plot:")],
        [sg.InputText(key='FILEPATH'), sg.FileSaveAs(
            'Save As', file_types=(("SVG Files", "*.svg"),))],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]

    window2 = sg.Window('Saving plot', layout)

    while True:
        event2, values2 = window2.read()

        if event2 in (sg.WIN_CLOSED, "Cancel"):
            window2.close()
            return

        if event2 == "Save":
            filename = values2.get('FILEPATH')
            filename_csv = filename+".csv"
            if filename:
                if not filename.endswith('.svg'):
                    filename += '.svg'
                plot.savefig(filename)
                data.to_csv(filename_csv)
                if stat_export:
                    # full stat report
                    output_file_path = f"{filename}_statistics.txt"
                    open(output_file_path, 'w').close()
                    with open(output_file_path, "a") as f:
                        print("Analysis carried out on:", file=f)
                        print(d1, file=f)
                        print("#################################", file=f)
                        print(test_result, file=f)
                        print(cld_result, file=f)
                window2.close()
                return
            else:
                sg.popup_error(
                    'Please choose a file location using Save As button.')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def draw_figure(canvas, figure):
    """
    Embeds a matplotlib figure in a Tkinter canvas.
    
    Parameters:
    canvas (Tk.Canvas): The Tkinter canvas widget to embed the figure in.
    figure (matplotlib.figure.Figure): The matplotlib figure to embed.
    
    The function performs the following steps:
    1. Creates a FigureCanvasTkAgg from the figure and canvas.
    2. Draws the figure onto the canvas.
    3. Packs the canvas widget into the Tkinter window.
    
    Returns:
    FigureCanvasTkAgg: The FigureCanvasTkAgg object created for the figure and canvas.
    
    Note:
    - This function is typically used in GUI applications where matplotlib plots need to be displayed in a Tkinter interface.
    """
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def plot_data_sf(df,window):
    """
    Creates a seaborn boxplot and stripplot from a DataFrame and displays it in a Tkinter canvas.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data to plot.

    The function performs the following steps:
    1. Creates a matplotlib subplot.
    2. Draws a seaborn boxplot and stripplot on the subplot using data from the DataFrame.
    3. Clears any existing plots from a specified Tkinter canvas.
    4. Embeds the new plot into the canvas using 'draw_figure'.
    5. Refreshes the window to update the display.

    Returns:
    matplotlib.figure.Figure: The figure object containing the generated plot.

    Note:
    - This function is designed for use in a GUI application with a Tkinter canvas element.
    """
    plot, ax = plt.subplots()  # Create a figure and an axes
    sns.boxplot(y="Average ratios", ax=ax, data=df)  # Draw boxplot on the axes
    sns.stripplot(y="Average ratios", ax=ax, data=df, jitter=True,
                  color='orange')  # Draw stripplot on the axes

    for item in window['-CANVAS2-'].TKCanvas.pack_slaves():
        item.destroy()

    draw_figure(window['-CANVAS2-'].TKCanvas, plot)
    window.refresh()
    return plot


def plot_data(df,window):
    """
    Creates a seaborn line plot from a DataFrame and displays it in a Tkinter canvas.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data to plot.

    The function performs the following steps:
    1. Creates a seaborn line plot with error bars using the DataFrame.
    2. Clears any existing plots from a specified Tkinter canvas.
    3. Embeds the new plot into the canvas using 'draw_figure'.
    4. Refreshes the window to update the display.

    Returns:
    matplotlib.figure.Figure: The figure object containing the generated plot.

    Note:
    - This function is designed for use in a GUI application with a Tkinter canvas element.
    """
    plot = sns.relplot(x="Time", y="Average", kind="line",
                       errorbar='se', data=df)
    for item in window['-CANVAS2-'].TKCanvas.pack_slaves():
        item.destroy()
    draw_figure(window['-CANVAS2-'].TKCanvas, plot.fig)
    window.refresh()
    return plot
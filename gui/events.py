# Event triggered code#########################################################

version = 5
firstfile = True
compiled_data = None
file_list = []  # A list to keep track of added files
plotcompiled = None
list_pattern_avoid = ["Drift-plot", "xyCorrected", "xyzCorrected"]
while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, "Cancel"):
        window.close()
        break

    if event == "Split":
        if values["-FOLDER0-"] != "":
            channel_splitter_czi(values["-FOLDER0-"])
        else:
            sg.popup_error('Please select a folder first')

    if event == "3D-Registration":
        if values["-FOLDER00-"] != "":
            folder = values["-FOLDER00-"]
            list_files = find_file_recursive(
                folder=folder, pattern=".tif")  # look for tif files only
            # extract filenames from full path
            list_filenames = [os.path.basename(path) for path in list_files]
            file_dict = dict(zip(list_filenames, list_files))
            list_fileC2 = [file for file in list(
                file_dict.keys()) if "C2" in file]
            list_fileC2 = [item for item in list_fileC2 if all(
                not re.search(pattern, item) for pattern in list_pattern_avoid)]
            list_fileC2 = [file_dict.get(filename) for filename in list_fileC2]
            list_fileC1 = [file for file in list(
                file_dict.keys()) if "C1" in file]
            list_fileC1 = [item for item in list_fileC1 if all(
                not re.search(pattern, item) for pattern in list_pattern_avoid)]
            list_fileC1 = [file_dict.get(filename) for filename in list_fileC1]

            if len(list_fileC2) == 0:
                sg.popup_error(
                    'Not C2 channel files detected, run SPLIT first')
            elif len(list_fileC2) == 0:
                sg.popup_error(
                    'Not C1 channel files detected, run SPLIT first')
            else:
                registration(referenceChannelFiles=sorted(
                    list_fileC2), applyToFiles=sorted(list_fileC1))
        else:
            sg.popup_error('Please select a folder first')
            
    if event == "Manual ROI selection":
        list_pattern_avoid = ["Drift-plot", "xyCorrected"]
        # retrieve list of image C2, ROI selection based on quality of PI staining
        if values["-FOLDER0000-"] != "":
            folder = values["-FOLDER0000-"]
            list_files = find_file_recursive(
                folder=folder, pattern=".tif")  # look for tif files only
            list_files_xyzCorrected = [
                file for file in list_files if "xyzCorrected" in file]
            # extract filenames from full path
            list_filenames = [os.path.basename(path) for path in list_files]
            file_dict = dict(zip(list_filenames, list_files))
            list_fileC2 = [file for file in list(
                file_dict.keys()) if "C2" in file]
            list_fileC2 = [item for item in list_fileC2 if all(
                not re.search(pattern, item) for pattern in list_pattern_avoid)]
            list_fileC2 = [file_dict.get(filename) for filename in list_fileC2]
            list_fileC1 = [file for file in list(
                file_dict.keys()) if "C1" in file]
            list_fileC1 = [item for item in list_fileC1 if all(
                not re.search(pattern, item) for pattern in list_pattern_avoid)]
            list_fileC1 = [file_dict.get(filename) for filename in list_fileC1]

            #import first movie in python then napari instance
            image2 = tifffile.imread(list_fileC2[0])
            viewer = napari.Viewer()
            viewer.add_image(image2)
            viewer.add_shapes()
            original_shape="empty"
            i=0 #increment for root list
            

            #Process ROI selection on PI channel
            @magicgui(call_button="Apply to all frames and slices")
            def applyShape(shapes_layer: napari.layers.Shapes,viewer: napari.Viewer):
                if len(shapes_layer.data) == 0:
                    warnings.warn("Please select ROI")
                    return
                global original_shape
                original_shape = shapes_layer.data[0]  # Get the first shape
                
                num_timeframes = int(viewer.dims.range[0][1])  # Total timeframes
                num_z_planes = int(viewer.dims.range[1][1])  # Total Z-planes
                
                new_shapes = []
                for t in range(num_timeframes):
                    for z in range(num_z_planes):
                        new_shape = original_shape.copy()
                        # Modify the shape's time and Z indices as needed
                        # This depends on how your shape's coordinates are structured
                        new_shapes.append(new_shape)
                
                shapes_layer.add(
                      new_shapes,
                      shape_type=['polygon'],
                      edge_width=5,
                      edge_color='coral',
                      face_color='orange'
                    )
                                    
            @magicgui(call_button="Process ROI")
            def process_roi(shapes_layer: napari.layers.Shapes, viewer: napari.Viewer):
                global i
                global original_shape
                try:
                    if type(original_shape)==str:
                        warnings.warn("Please select ROI and apply")
                        return
                    if original_shape.shape[1]>2:
                        roi_shape = original_shape[:, 2:4]
                    else:
                        roi_shape = original_shape 
                    current_z = viewer.dims.current_step[1]  # Current Z index
            
                    # Extract the x and y coordinates from the shape data (assuming they are in the 3rd and 4th columns)
                    roi_vertices = roi_shape
            
                    # Determine the bounding box (min and max coordinates) of the ROI
                    min_y, max_y = int(np.min(roi_vertices[:, 0]))+1, int(np.max(roi_vertices[:, 0]))+1
                    min_x, max_x = int(np.min(roi_vertices[:, 1]))+1, int(np.max(roi_vertices[:, 1]))+1

            
                    # Create a path from the ROI vertices
                    path = Path(roi_vertices)
            
                    # Process each frame at the current Z
                    processed_stack = []
                    for t_frame in viewer.layers[0].data[:, current_z, :, :]:
                        # Crop the frame to the bounding box of the ROI
                        cropped_frame = t_frame[min_y:max_y,min_x:max_x]
            
                        # Create a mask for the cropped area
                        cropped_mask = path.contains_points(np.c_[np.mgrid[min_y:max_y, min_x:max_x].reshape(2, -1).T]).reshape(cropped_frame.shape)
            
                        # Apply the mask to the cropped frame
                        processed_frame = np.where(cropped_mask, cropped_frame, 0)
                        processed_stack.append(processed_frame)
                        
                    # Add the processed stack as a new layer
                    viewer.add_image(np.array(processed_stack), name="Processed Stack")
                    
                    #Process biosensor data
                    image2 = tifffile.imread(list_fileC1[0])
                    processed_stack_biosensor = []
                    for t_frame in image2[:, current_z, :, :]:
                        # Crop the frame to the bounding box of the ROI
                        cropped_frame = t_frame[min_y:max_y,min_x:max_x]
            
                        # Create a mask for the cropped area
                        cropped_mask = path.contains_points(np.c_[np.mgrid[min_y:max_y, min_x:max_x].reshape(2, -1).T]).reshape(cropped_frame.shape)
            
                        # Apply the mask to the cropped frame
                        processed_frame = np.where(cropped_mask, cropped_frame, 0)
                        processed_stack_biosensor.append(processed_frame)
                    
                    #save only save when next root is clicked!!!
                    path_pi=os.path.dirname(list_fileC2[i])+"/pi.tif"
                    tifffile.imwrite(path_pi,processed_stack)
                    path_fluo=os.path.dirname(list_fileC1[i])+"/fluo.tif"
                    tifffile.imwrite(path_fluo,processed_stack_biosensor)
                    
                    original_shape="empty" #reinitialize shape
                except Exception as e:
                    warnings.warn(f"An error occurred: {e}")
            
            @magicgui(call_button="Next")
            def nextMovie(shapes_layer: napari.layers.Shapes, viewer: napari.Viewer):
                global i
                i+=1
                if i > len(list_fileC2):
                    warnings.warn("No more movies to process")
                    container.close()
                    QTimer.singleShot(100, close_viewer_safely)
                    return
                    
                    return
                else:
                    viewer.layers.clear()
                    image2 = tifffile.imread(list_fileC2[0])
                    viewer.add_image(image2)
                    viewer.add_shapes()
                    global original_shape
                    original_shape="empty"
                
 
            # Add the process_roi function as a widget
            container = widgets.Container()
            # Add buttons to the container
            container.append(applyShape)
            container.append(process_roi)
            container.append(nextMovie)
            
            # Add the container as a dock widget to the viewer
            viewer.window.add_dock_widget(container, area='right')
            viewer.show()
            app.exec_()
            
            
            #selection to all frames in this z
            #crop and clear
            #next root (save last and open new one)
            
    if event == "Tracer-analysis":
        list_pattern_avoid = ["Drift-plot", "xyCorrected"]
        # retrieve list of image C1
        if values["-FOLDER000-"] != "":
            folder = values["-FOLDER000-"]
            list_files = find_file_recursive(
                folder=folder, pattern=".tif")  # look for tif files only
            list_files_xyzCorrected = [
                file for file in list_files if "xyzCorrected" in file]
            # extract filenames from full path
            list_filenames = [os.path.basename(path) for path in list_files]
            file_dict = dict(zip(list_filenames, list_files))
            list_fileC1 = [file for file in list(
                file_dict.keys()) if "C1" in file]
            list_fileC1 = [item for item in list_fileC1 if all(
                not re.search(pattern, item) for pattern in list_pattern_avoid)]
            list_fileC1 = [file_dict.get(filename) for filename in list_fileC1]
            
        for file in list_fileC1:
            data = tifffile.imread(file)  # import full stabilized C1 file
            mip_data = np.median(data, axis=1)  # median Z projection
            # Loop through each timeframe and calculate the corner averages
            corner_avgs = []
            for i in range(mip_data.shape[0]):
                avg = corner_average(mip_data[i])
                corner_avgs.append(avg)
            corner_avgs = np.array(corner_avgs)
            df = pd.DataFrame(
                {'Timeframe': range(len(corner_avgs)), 'Corner Average': corner_avgs})

            # Calculate the first-order difference
            diffs = np.diff(corner_avgs)

            # Compute mean and standard deviation of differences
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)

            # Identify significant changes (more than 2 std_devs from the mean)
            threshold = mean_diff + 2 * std_diff
            significant_changes = np.where(np.abs(diffs) > threshold)[0]
            image_parameters = [f'Number of timeframes:{data.shape[0]}',
                                f'Number of slices:{data.shape[1]}',
                                f'Dimension x:{data.shape[2]}',
                                f'Dimension y:{data.shape[3]}',
                                f'Tracer significant change at frame:{significant_changes[0]}']
            # Split each string into name and value
            split_parameters = [param.split(":") for param in image_parameters]

            # Create DataFrames and save
            df.to_csv(f'{file}_tracer_measured.csv')
            df_im = pd.DataFrame(split_parameters, columns=[
                                 "Parameter", "Value"])
            df_im.to_csv(f'{file}_parameters.csv')

            # Save tracer plot
            # Create the figure and axes objects
            tracer_figure, ax = plt.subplots(figsize=(10, 6))

            # Create the plot on the Axes object
            ax.plot(corner_avgs, label='Corner Averages')

            # Add significant changes
            for change in significant_changes:
                ax.axvline(change, color='g', linestyle='--',
                           label=f'Significant Change at {change}')

            # Add labels and title
            ax.set_xlabel('Timeframe')
            ax.set_ylabel('Corner Average')
            ax.set_title('Corner Averages Over Time with Significant Changes')

            # Add legend
            ax.legend()

            # Save the plot
            tracer_figure.savefig(f'{file}_tracer_averages_plot.png')

    if event == "Plot":
        final_df, plotcompiled = plot_data_compiled()

    if event == "Save plot to":
        if plotcompiled:
            save_compiled_plot(data=final_df, plot=plotcompiled)
        else:
            sg.popup_error('Plot before saving maybe?')
    # iterator for number of cells in variable replacement

    if event == "Normalize and plot":

        final_df, plotcompiled = plot_data_compiled_norm()

    if event == "Add":
        # Extract the file name from the full path

        file_name = values['File']
        if file_name.endswith('.csv'):
            if "results" in file_name:  # movie processesing
                data = process_file(file_name)
                window['File'].update(value='')
                file_name_only = os.path.basename(values['File'])
                file_list.append(file_name_only)  # Add the file to the list
                # Display the file list
                window['-CONSOLE2-'].update('\n'.join(file_list))
                window['File'].update(value='')  # Clear the file input field
                # Handling the data storage based on whether it's the first file or subsequent files
                if firstfile:
                    # z=0
                    compiled_data = data
                    # compiled_data,ite=dfreplace(df=compiled_data,variable="variable",iterator=z)
                    # z+=ite
                    firstfile = False
                else:
                    # data,ite=dfreplace(df=data,variable="variable",iterator=z)
                    # z+=ite
                    compiled_data = pd.concat([compiled_data, data], axis=0)

                    compiled_data = compiled_data.reset_index(drop=True)
            else:  # single frame processing
                data = process_file_sf(file_name)
                window['File'].update(value='')
                file_name_only = os.path.basename(values['File'])
                file_list.append(file_name_only)  # Add the file to the list
                # Display the file list
                window['-CONSOLE2-'].update('\n'.join(file_list))
                window['File'].update(value='')  # Clear the file input field

                if firstfile:
                    compiled_data = data
                    firstfile = False
                else:
                    compiled_data = pd.concat([compiled_data, data], axis=0)
                    compiled_data = compiled_data.reset_index(drop=True)

        else:
            sg.popup_error('Please select a valid CSV file.')
            
    if event == "Clear list":
        window['-CONSOLE2-'].update('')
        window['File'].update(value='')
        compiled_data=None
    
    if event == "Compile":
        if compiled_data is not None:
            if "Time" not in compiled_data.columns:  # single frame plotting
                plot = plot_data_sf(compiled_data)
                get_save_folder(data=compiled_data, plot=plot)
                window['-CONSOLE2-'].update('')  # Clear the multiline element
                window['File'].update(value='')  # Clear the file input field

            else:  # movies plotting
                plot = plot_data(compiled_data)
                get_save_folder(data=compiled_data, plot=plot)
                window['-CONSOLE2-'].update('')  # Clear the multiline element
                window['File'].update(value='')  # Clear the file input field
                z = 0
        else:
            sg.popup_error("No valid data to process.")
        file_list = []
        compiled_data = None

    if event == 'Clear':
        window['-FOLDER1-'].update(value='')
        window['-FOLDER2-'].update(value='')
        window['-FOLDER3-'].update(value='')
        window['-FOLDER4-'].update(value='')

    if event == "Run image processing Single frame":
        if values["erosion"] == "564":
            erosionfactor = 5
        elif values["erosion"] == "991":
            erosionfactor = 1
        elif values["erosion"] == "604":
            erosionfactor = 3

        biosensor_folder = values["-FOLDER12-"]
        pi_folder = values["-FOLDER22-"]
        if biosensor_folder == "" or pi_folder == "":
            sg.popup_error(
                "One the folders required for the analysis is not specified", title="Folder error")

        # retrieve image list
        biosensor_img = find_file(folder=biosensor_folder, pattern=".tif")
        biosensor_img.sort()
        # retrieve image pi list
        pi_img = find_file(folder=pi_folder, pattern=".tif")
        pi_img.sort()
        # check same number of images and names
        if len(biosensor_img) != len(pi_img):
            sg.popup_error(
                "Images for biosensor and pi are not matching in numbers", title="file error")

        biosensor_img_base = [os.path.basename(file) for file in biosensor_img]
        biosensor_img_base = [file.replace(
            file[0:3], "") for file in biosensor_img_base]
        biosensor_img_base.sort()
        pi_img_base = [os.path.basename(file) for file in pi_img]
        pi_img_base = [file.replace(file[0:3], "") for file in pi_img_base]
        pi_img_base.sort()
        
        output_compare = compareList(l1=biosensor_img_base, l2=pi_img_base)
        if output_compare == "Non equal":
            sg.popup_error("Image names are not matching", title="file error")

        # process each binome of file one by one
        i = 0
        try:
            for img_pi_path, img_biosensor_path in zip(pi_img,biosensor_img):
                img_nb = len(biosensor_img)
                update_console(
                    window, "-CONSOLE1-", f'Analysing couple of images number {i+1}/{img_nb}')
                image_pi = tifffile.imread(img_pi_path)
                image_biosensor = tifffile.imread(img_biosensor_path)
                update_console(window, "-CONSOLE1-",'generating segmentation')
                membranes, novacuole, intracellular = segmentation_all(image_pi, image_biosensor)
                

                # plot
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

                # Measure mean fluorescences and positions
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
                
                i += 1
            # make table ratio with label, mean mb, mean cyt, ratio, filename

        except Exception as e:
            update_console(
                window, "-CONSOLE1-", f"An error occurred while processing folder number {img_biosensor_path}: {str(e)}")
            update_console(window, "-CONSOLE1-",
                           "Continuing with the next folder.")

        sg.popup_no_frame('Image analysis is done!')

    if event == "Run image processing":
        if values["erosion"] == "564":
            erosionfactor = 5
        elif values["erosion"] == "991":
            erosionfactor = 1
        elif values["erosion"] == "604":
            erosionfactor = 3
        # determine how many folder were specified by user
        nb_folders_toanalyse = 0
        for folder_nb in np.arange(1, 5, 1):
            folder_path = values["-FOLDER"+str(folder_nb)+"-"]
            if len(folder_path) > 0:
                nb_folders_toanalyse += 1
        try:
            for folder_nb in np.arange(1, nb_folders_toanalyse+1, 1):
                folder_path = values["-FOLDER"+str(folder_nb)+"-"]
                savename = os.path.basename(folder_path)
                update_console(
                    window, "-CONSOLE-", f'Analysing folder number {str(folder_nb)}/{str(nb_folders_toanalyse)}')
                # Do something with the selected folder path
                directory = folder_path
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
                    update_console(
                        window, "-CONSOLE-", f'generating cytosol and membrane masks frame n° {str(i)}')
                    # if values["method"]=="method 1":
                    # Median blur denoising
                    denoised_image = cle.median_box(
                        frame, radius_x=10, radius_y=10, radius_z=0)
                    # Maximum filter
                    denoised_image2 = cle.fabs(denoised_image)
                    denoised_image2 = ndimage.maximum_filter(
                        cle.pull(denoised_image2), size=5)
                    # Top hat
                    denoised_image2 = cle.top_hat_box(
                        denoised_image2, radius_x=20, radius_y=20, radius_z=0)
                    # Sqrt filter
                    denoised_image3 = cle.sqrt(denoised_image2)

                    # Otsu auto threshold
                    binary1 = cle.threshold_otsu(denoised_image3)
                    # Closing operation to fill gaps
                    binary = cle.closing_labels(binary1, radius=2)
                    skeleton = skeletonize(cle.pull(binary))
                    skeleton = (skeleton > 0).astype(np.uint8) * 255
                    pruned_skeleton, segmented_img, segment_objects = prune(
                        skel_img=skeleton, size=1000)
                    cle_image = cle.push(pruned_skeleton)
                    dilate = cle.dilate_labels(cle_image, cle_image, radius=20)
                    dilate = cle.closing_labels(dilate, radius=15)
                    dilate = cle.erode_labels(dilate, radius=15)
                    dilate = skeletonize(cle.pull(dilate))
                    cle_image = cle.push(dilate)
                    dilate = cle.dilate_labels(cle_image, cle_image, radius=3)
                    dilate = cle.closing_labels(dilate, radius=12)
                    skeleton_stack.append(cle.pull(dilate))

                    inverted = np.asarray(dilate) == 0 * 1
                    label = cle.connected_components_labeling_box(inverted)
                    exclude = cle.exclude_labels_on_edges(label)
                    exclude = cle.exclude_labels_outside_size_range(
                        exclude, minimum_size=2500, maximum_size=100000000)

                    # Vacuole mask from sensor fluorescence
                    update_console(window, "-CONSOLE-",
                                   f'generating vacuole masks')
                    denoised_image = cle.median_box(
                        frameps, radius_x=5, radius_y=5, radius_z=0)
                    mini = cle.minimum_box(
                        denoised_image, radius_x=5, radius_y=5, radius_z=0)
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
                    denoised_image1 = cle.gaussian_blur(
                        frameps, sigma_x=2, sigma_y=2)
                    denoised_image2 = cle.median_box(
                        frameps, radius_x=2, radius_y=2)
                    cyt_one = cle.divide_images(cytcorrected, cytcorrected)
                    # 5 for 564, 3 for 604? and 1 for 991
                    cyt_one = cle.erode_labels(cyt_one, radius=erosionfactor)
                    cyt_one = cle.multiply_images(cyt_one, denoised_image2)
                    denoised_image3 = cle.top_hat_box(
                        cyt_one, radius_x=10, radius_y=10)
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
                    draw_figure(window['-CANVAS-'].TKCanvas, fig)
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
                if values["save_stitched"]:
                    imsave(directory+'/stitchimage.tif', stitched_image)
                # ordering table
                table_mb = table_mb.sort_values(by=['label', "Frames"])
                table_cyt = table_cyt.sort_values(by=['label', "Frames"])

                # Checking cells number in each frames
                # determine how cells on 1st frame
                # -1 because background pixel =0 is counting
                ncells = len(np.unique(cytosol_stack[0]))-1
                i = 0
                list_frames_tocorrect = []
                for frame in cytosol_stack:
                    ncells_temp = len(np.unique(frame))-1
                    if ncells_temp != ncells:
                        list_frames_tocorrect.append(i)

                    i += 1

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

                    # table_cyt.to_csv('/home/adminelson/Documents/FERONIA_RALFs/Imaging definitive conditions/ralf23/564/20230418-564-ralf23-4microM/Experiment-404-Airyscan Processing-05/C2-Experiment-408-Airyscan Processing-05.czi - Experiment-408-Airyscan Processing-05.czi #1_2023-Apr-21-001/cyt.csv')
                    # table_mb.to_csv('/home/adminelson/Documents/FERONIA_RALFs/Imaging definitive conditions/ralf23/564/20230418-564-ralf23-4microM/Experiment-404-Airyscan Processing-05/C2-Experiment-408-Airyscan Processing-05.czi - Experiment-408-Airyscan Processing-05.czi #1_2023-Apr-21-001/mb.csv')
                else:
                    update_console(
                        window, "-CONSOLE-", f'!!!! Error detected in cell labelling, correction in progress !!!!')
                    table_mb2 = table_mb.copy()
                    table_cyt2 = table_cyt.copy()
                    for wrong_frame in list_frames_tocorrect:
                        update_console(window, "-CONSOLE-",
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
                      # add line for missing frames for some object maybe after reordering first
                    ntf_ref = len(cytosol_stack)
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
                    # table_cyt2.to_csv('/home/adminelson/Documents/FERONIA_RALFs/Imaging definitive conditions/ralf23/564/20230418-564-ralf23-4microM/Experiment-404-Airyscan Processing-05/C2-Experiment-408-Airyscan Processing-05.czi - Experiment-408-Airyscan Processing-05.czi #1_2023-Apr-21-001/cyt_corrected.csv')
                    # table_mb2.to_csv('/home/adminelson/Documents/FERONIA_RALFs/Imaging definitive conditions/ralf23/564/20230418-564-ralf23-4microM/Experiment-404-Airyscan Processing-05/C2-Experiment-408-Airyscan Processing-05.czi - Experiment-408-Airyscan Processing-05.czi #1_2023-Apr-21-001/mb_corrected.csv')
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

                # correct frame negative/positive or control/treatment according to parameters file
                # open file

                img_parameters = pd.read_csv(
                    find_file(folder_path, "_parameters.csv")[0])
                # retrieve significant change value
                change = img_parameters["Value"][4]+1

                # modify frame columns
                df_ratio.index = df_ratio.index-change
                df_ratio.to_csv(directory+'/results-LRC'+str(version) +
                                '-erode_'+str(erosionfactor)+"-"+date_str+savename+'.csv')

        except Exception as e:
            update_console(
                window, "-CONSOLE-", f"An error occurred while processing folder number {folder_nb}: {str(e)}")
            update_console(window, "-CONSOLE-",
                           f"Continuing with the next folder.")

        update_console(window, "-CONSOLE-", f"DONE!!")
        sg.popup_no_frame('Image analysis is done!')
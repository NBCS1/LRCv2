import tifffile
import napari

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
from utils.image_analyses import parameterfile
def napariROI(list_fileC2,list_fileC1,app):
    
    image2 = tifffile.imread(list_fileC2[0])
    viewer = napari.Viewer()
    viewer.add_image(image2)
    viewer.add_shapes()
    original_shape="empty"
    i=0 #increment for root list
    napari.utils.notifications.show_info("Please select a ROI using the Polygon selection")
    
    #Process ROI selection on PI channel
    @magicgui(call_button="Apply to all frames and slices")
    def applyShape(shapes_layer: napari.layers.Shapes,viewer: napari.Viewer):
        if len(shapes_layer.data) == 0:
            warnings.warn("Please select ROI")
            return
        nonlocal original_shape
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
        napari.utils.notifications.show_info("If the ROI good for you, press Process ROI")
                            
    @magicgui(call_button="Process ROI")
    def process_roi(shapes_layer: napari.layers.Shapes, viewer: napari.Viewer):
        napari.utils.notifications.show_info("Cropping of movies on both channel on going, please wait, I'm not frozen")
        nonlocal original_shape
        nonlocal i
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
            image2 = tifffile.imread(list_fileC1[i])
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
            parameterfile(image2,path=list_fileC1[i])
            original_shape="empty" #reinitialize shape
            viewer.layers[0].visible = False
            viewer.layers[1].visible = False
        except Exception as e:
            warnings.warn(f"An error occurred: {e}")
    
    @magicgui(call_button="Next")
    def nextMovie(shapes_layer: napari.layers.Shapes, viewer: napari.Viewer):
        nonlocal i
        
        i+=1
        if i > len(list_fileC2)-1:
            warnings.warn("No more movies to process")
            container.close()
            def close_viewer_safely():
                """
                Closes an image viewer application safely.

                The function checks if the viewer application is open and closes it, ensuring that the associated application instance is also terminated.

                Note:
                - The function assumes the presence of a global 'viewer' variable representing the viewer application.
                - It is designed for use with applications that require explicit termination of the application instance.
                """
                if viewer:
                    viewer.close()
                    QApplication.instance().quit()     
            QTimer.singleShot(100, close_viewer_safely)
            return
            
        else:
            viewer.layers.clear()
            image2 = tifffile.imread(list_fileC2[i])
            viewer.add_image(image2)
            viewer.add_shapes()
            nonlocal original_shape
            original_shape="empty"
        
    
    @magicgui(call_button="Pass ROI selection process")
    def passProcessing(viewer: napari.Viewer):
        def close_viewer_safely():
            if viewer:
                viewer.close()
                QApplication.instance().quit()     
        QTimer.singleShot(100, close_viewer_safely)
        return
    # Add the process_roi function as a widget
    container = widgets.Container()
    # Add buttons to the container
    container.append(applyShape)
    container.append(process_roi)
    container.append(nextMovie)
    container.append(passProcessing)
    
    # Add the container as a dock widget to the viewer
    viewer.window.add_dock_widget(container, area='right',name="ROI selection")
    viewer.show()
    app.exec_()
    
def napariROI_single(list_fileC2,list_fileC1,app):
    
    image2 = tifffile.imread(list_fileC2[0])
    viewer = napari.Viewer()
    viewer.add_image(image2)
    viewer.add_shapes()
    original_shape="empty"
    i=0 #increment for root list
    
                            
    @magicgui(call_button="Process ROI")
    def process_roi(shapes_layer: napari.layers.Shapes, viewer: napari.Viewer):
        original_shape=viewer.layers[1]
        nonlocal i
        try:

            
            roi_shape = original_shape.data

            # Extract the x and y coordinates from the shape data (assuming they are in the 3rd and 4th columns)
            roi_vertices = roi_shape[0]
    
            # Determine the bounding box (min and max coordinates) of the ROI
            min_y, max_y = int(np.min(roi_vertices[:, 0])) + 1, int(np.max(roi_vertices[:, 0])) + 1
            min_x, max_x = int(np.min(roi_vertices[:, 1])) + 1, int(np.max(roi_vertices[:, 1])) + 1
    
            # Create a path from the ROI vertices
            path = Path(roi_vertices)
    
            # Process each frame at the current Z
            processed_stack = []
            # Crop the frame to the bounding box of the ROI
            cropped_frame = viewer.layers[0].data[min_y:max_y,min_x:max_x]

            # Create a mask for the cropped area
            cropped_mask = path.contains_points(np.c_[np.mgrid[min_y:max_y, min_x:max_x].reshape(2, -1).T]).reshape(cropped_frame.shape)

            # Apply the mask to the cropped frame
            processed_frame = np.where(cropped_mask, cropped_frame, 0)
            processed_stack.append(processed_frame)
    
            # Add the processed stack as a new layer
            viewer.add_image(np.array(processed_stack), name="Processed Stack")
    
            # Process biosensor data
            image2 = tifffile.imread(list_fileC1[i])
            processed_stack_biosensor = []
            # Crop the frame to the bounding box of the ROI
            cropped_frame = image2[min_y:max_y,min_x:max_x]

            # Create a mask for the cropped area
            cropped_mask = path.contains_points(np.c_[np.mgrid[min_y:max_y, min_x:max_x].reshape(2, -1).T]).reshape(cropped_frame.shape)

            # Apply the mask to the cropped frame
            processed_frame = np.where(cropped_mask, cropped_frame, 0)
            processed_stack_biosensor.append(processed_frame)
    
            # Save processed data in a new folder "/processed/"
            image_folder=os.path.dirname(list_fileC2[i])
            processed_folder=image_folder+"/processed/"
            if not os.path.exists(processed_folder): 
                os.makedirs(processed_folder) 
                
            path_pi = processed_folder+os.path.basename(list_fileC2[i]) + "_pi.tif"
            tifffile.imwrite(path_pi, processed_stack)
            path_fluo = processed_folder+os.path.basename(list_fileC1[i]) + "_fluo.tif"
            tifffile.imwrite(path_fluo, processed_stack_biosensor)
    
            original_shape = "empty"  # Reinitialize shape
    
        except Exception as e:
            warnings.warn(f"An error occurred: {e}")


    
    @magicgui(call_button="Next")
    def nextMovie(shapes_layer: napari.layers.Shapes, viewer: napari.Viewer):
        nonlocal i
        i+=1
        if i > len(list_fileC2)-1:
            warnings.warn("No more movies to process")
            container.close()
            def close_viewer_safely():
                """
                Closes an image viewer application safely.

                The function checks if the viewer application is open and closes it, ensuring that the associated application instance is also terminated.

                Note:
                - The function assumes the presence of a global 'viewer' variable representing the viewer application.
                - It is designed for use with applications that require explicit termination of the application instance.
                """
                if viewer:
                    viewer.close()
                    QApplication.instance().quit()     
            QTimer.singleShot(100, close_viewer_safely)
            return
            
        else:
            viewer.layers.clear()
            image2 = tifffile.imread(list_fileC2[i])
            viewer.add_image(image2)
            viewer.add_shapes()
            nonlocal original_shape
            original_shape="empty"
        
    @magicgui(call_button="Pass ROI selection process")
    def passProcessing(viewer: napari.Viewer):
        def close_viewer_safely():
            if viewer:
                viewer.close()
                QApplication.instance().quit()     
        QTimer.singleShot(100, close_viewer_safely)
        return
    # Add the process_roi function as a widget
    container = widgets.Container()
    # Add buttons to the container
    container.append(process_roi)
    container.append(nextMovie)
    container.append(passProcessing)
    
    # Add the container as a dock widget to the viewer
    viewer.window.add_dock_widget(container, area='right',name="Single Frame ROI selection")
    viewer.show()
    app.exec_()
    
from utils.image_analyses import tracerAnalysis
def napariTracer(list_fileC1,app):
    
    image2 = tifffile.imread(list_fileC1[0])
    viewer2 = napari.Viewer()
    viewer2.add_image(image2)
    viewer2.add_shapes()
    original_shape="empty"
    i=0 #increment for root list
    
    
    #Process ROI selection on PI channel
    @magicgui(call_button="Quantify tracer and Next")
    def Quantifytracer(shapes_layer: napari.layers.Shapes,viewer2: napari.Viewer):
        nonlocal i
        nonlocal original_shape
        if len(shapes_layer.data) == 0:
            warnings.warn("Please select ROI")
            return
        nonlocal original_shape
        original_shape = viewer2.layers[1].data[0]  # Get the first shape
        print(original_shape)
        
        try:
            if type(original_shape)==str:
                warnings.warn("Please select ROI and apply")
                return
            if original_shape.shape[1]>2:
                roi_shape = original_shape[:, 2:4]
            else:
                roi_shape = original_shape 
    
            # Extract the x and y coordinates from the shape data (assuming they are in the 3rd and 4th columns)
            roi_vertices = roi_shape
    
            # Determine the bounding box (min and max coordinates) of the ROI
            min_y, max_y = int(np.min(roi_vertices[:, 0]))+1, int(np.max(roi_vertices[:, 0]))+1
            min_x, max_x = int(np.min(roi_vertices[:, 1]))+1, int(np.max(roi_vertices[:, 1]))+1
    
    
            # Create a path from the ROI vertices
            path = Path(roi_vertices)
    
            # Process each frame and Z to crop
            processed_stack = []

            # Iterate over all time frames
            for t_frame in viewer2.layers[0].data:
                # Iterate over all Z slices
                for z in range(t_frame.shape[0]):
                    # Crop the frame to the bounding box of the ROI
                    cropped_frame = t_frame[z, min_y:max_y, min_x:max_x]
            
                    # Create a mask for the cropped area
                    cropped_mask = path.contains_points(np.c_[np.mgrid[min_y:max_y, min_x:max_x].reshape(2, -1).T]).reshape(cropped_frame.shape)
            
                    # Apply the mask to the cropped frame
                    processed_frame = np.where(cropped_mask, cropped_frame, 0)
                    
                    # Append the processed frame to the stack
                    processed_stack.append(processed_frame)
            processed_stack = np.array(processed_stack).reshape(viewer2.layers[0].data.shape[0],viewer2.layers[0].data.shape[1],cropped_mask.shape[0],cropped_mask.shape[1])
            print("Stack has been cropped out")
            tracerAnalysis(processed_stack,list_fileC1[i])
            
        except Exception as e:
            warnings.warn(f"An error occurred: {e}")
    
        i+=1
        if i > len(list_fileC1)-1:
            warnings.warn("No more movies to process")
            container.close()
            def close_viewer_safely():
                """
                Closes an image viewer application safely.

                The function checks if the viewer application is open and closes it, ensuring that the associated application instance is also terminated.

                Note:
                - The function assumes the presence of a global 'viewer' variable representing the viewer application.
                - It is designed for use with applications that require explicit termination of the application instance.
                """
                if viewer2:
                    viewer2.close()
                    QApplication.instance().quit()     
            QTimer.singleShot(100, close_viewer_safely)
            return
            
        else:
            viewer2.status = "Next movie is loading"
            viewer2.layers.clear()
            image2 = tifffile.imread(list_fileC1[i])
            viewer2.add_image(image2)
            viewer2.add_shapes()
            original_shape="empty"
        
    
    # Add the process_roi function as a widget
    container = widgets.Container()
    # Add buttons to the container
    container.append(Quantifytracer)
    
    # Add the container as a dock widget to the viewer
    viewer2.window.add_dock_widget(container, area='right',name="Background selection")
    viewer2.show()
    app.exec_()
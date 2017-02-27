import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'BGR':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, xpointl = 0, xpointr = 0, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
    	x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    if (xpointl == 0):
	    xpointl = x_start_stop[0]
    if (xpointr == 0):
	    xpointr = x_start_stop[1]
	# If x and/or y start/stop positions not defined, set to image size
	
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    inc_x_pix = (xpointl - x_start_stop[0])//nx_windows
    dec_x_pix = (x_start_stop[1] - xpointr)//nx_windows
    out_of_bounds = 0
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    beginx = x_start_stop[0]
    endx = x_start_stop[1]
    for ys in range(ny_windows):
        beginx = (ny_windows-ys+1) * inc_x_pix 
        #beginx = beginx
        endx = img.shape[1]
        nx_windows = np.int((xspan+beginx)/nx_pix_per_step) - 1#np.int((endx-beginx)/nx_pix_per_step) - 1
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0] + beginx
            finishx = startx + xy_window[0]
            if finishx > img.shape[1]:
                finishx = img.shape[1]
                startx = finishx - xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                finishy = starty + xy_window[1]
                window_list.append(((startx, starty), (finishx, finishy)))
                out_of_bounds += 1
                break

            starty = ys*ny_pix_per_step + y_start_stop[0]
            finishy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (finishx, finishy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
    # Iterate through the bounding boxes
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
		#bboxcenterx = (bbox[0,0] + bbox[1,0])/2
		#bboxcentery = (bbox[0,1] + bbox[1,1])/2
		#print("x, y = ", bboxcenterx, bboxcentery)
		#if (bbox[0,0] - bbox[1,0]) < 100:
		#	bbox[0,0] = (bbox[1,0] + bbox[0,0])/2 -50
		#	bbox[1,0] = bbox[0,0] + 100
		#if ((bbox[0,1] - bbox[1,1]) < 150):
		#	bbox[0,1] = (bbox[1,1] + bbox[0,1])/2 -75
		#	bbox[1,1] = bbox[0,1] + 150
		
		# Draw a rectangle given bbox coordinates

		#cv2.rectangle(imcopy, (bboxcenterx-75, bboxcentery-50), (bboxcenterx+75, bboxcentery+50), color, thick)
	return imcopy
    # Return the image copy with boxes drawn
def get_windows(img, windowx , windowy, x_point_l, x_point_r, x_start_stop = [None,None], y_start_stop = [None,None],  
                num_sizes = 8, dec_per = 0.85, xy_overlap = (0.0,0), vis = False):
    x_ROI = x_start_stop
    y_ROI = y_start_stop
    inc_x_pix = (x_point_l - x_ROI[0])//num_sizes
    #dec_x_pix = (x_ROI[1]-x_point_r)//num_sizes
    dec_x_pix = 0
    window_list = []
    
    if x_ROI[0] == None:
        x_ROI[0] = 0
    if x_ROI[1] == None:
        x_ROI[1] = img.shape[1]
    if y_ROI[0] == None:
        y_ROIp[0] = 0
    if y_ROI[1] == None:
        y_ROI[1] = img.shape[0]
    for num in range(num_sizes):
        xspan = x_ROI[1] - x_ROI[0]
        yspan = y_ROI[1] - y_ROI[0]
        x_step_range = np.int(windowx*(1 - xy_overlap[0]))
        num_windows = np.int(xspan/x_step_range) #- 1
        window_vis = []
        for x in range(num_windows):
            startx = x*x_step_range + x_ROI[0]
            endx = startx + windowx                    
            starty = y_ROI[0]
            endy = starty + windowy
            if endx < x_ROI[1]:
                window_list.append(((startx, starty), (endx, endy)))
                window_vis.append(((startx, starty), (endx, endy)))
        window_list.append(((x_ROI[1]-windowx, starty), (x_ROI[1], endy)))
        window_vis.append(((x_ROI[1]-windowx, starty), (x_ROI[1], endy)))
        if vis == True:
            image_loc = 'test_images/test' + str(1) + '.jpg'
            image = mpimg.imread(image_loc)
            draw_img = draw_boxes(image, window_vis)
            plt.figure(num+2)
            plt.imshow(draw_img)
        #change window size
        x_ROI[1] = x_ROI[1] - dec_x_pix  
        x_ROI[0] = x_ROI[0] + inc_x_pix
        print('window size = ', windowx, 'x ', windowy)
        windowx = int(windowx * dec_per)
        windowy = int(windowy * dec_per)
        y_ROI[1] = int(y_ROI[1] * dec_per)
    for z in range(14):
            startx = z*64 + 320
            endx = startx + 128                    
            starty = 518
            endy = 646
            window_list.append(((startx, starty), (endx, endy)))
            window_vis.append(((startx, starty), (endx, endy)))
            if endx < x_ROI[1]:
                window_list.append(((startx, starty), (endx, endy)))
                window_vis.append(((startx, starty), (endx, endy)))    

    return window_list

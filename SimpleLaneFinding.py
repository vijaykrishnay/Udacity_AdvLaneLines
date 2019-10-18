import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def grayscale(img, is_RGB=True):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    if is_RGB:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        # use BGR2GRAY if you read an image with cv2.imread()
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def calc_line_x(y, slope, intercept):
    """Calculate x value given y, slope, intercept"""
    return int((y - intercept)/slope)

def merge_lane_lines(lines, image_height, slope_thd=0.5):
    """
    1. Separate lines to left, right lanes based on slope. 
    2. Calculated avg. slope, intercept weighted by line length
    3. Calculate left, right lane end points based on above slopes, image height
    """
    
    left_y_top = right_y_top = image_height
    left_slope = right_slope = 0
    left_intercept = right_intercept = 0
    left_weight = right_weight = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 == x1:
                continue

            m = (y2-y1)/(x2-x1)
            c = y1 - m * x1
            d = ((y2-y1)**2 + (x2-x1)**2)**0.5

            # Update left, right lane weighted slopes, intercepts
            if m > slope_thd:
                # Left Lane Lines
                left_slope = (left_slope * left_weight + m * d)/(left_weight + d)
                left_intercept = (left_intercept * left_weight + c * d)/(left_weight + d)
                left_weight += d
                left_y_top = min(left_y_top, y1, y2)
            elif m < -slope_thd:
                # Right Lane Lines
                right_slope = (right_slope * right_weight + m * d)/(right_weight + d)
                right_intercept = (right_intercept * right_weight + c * d)/(right_weight + d)
                right_weight += d
                right_y_top = min(right_y_top, y1, y2)

    # Use same value for left, right y top (lines are parallel!)
    left_y_top = right_y_top = min(left_y_top, right_y_top)
                
    # Calculate left, right lane end points
    left_y_bottom = right_y_bottom = image_height
    left_x_top = calc_line_x(left_y_top, slope=left_slope, intercept=left_intercept)
    right_x_top = calc_line_x(right_y_top, slope=right_slope, intercept=right_intercept)
    left_x_bottom = calc_line_x(left_y_bottom, slope=left_slope, intercept=left_intercept)
    right_x_bottom = calc_line_x(right_y_bottom, slope=right_slope, intercept=right_intercept)

    left_line_merged = np.array([[left_x_top, left_y_top, left_x_bottom, left_y_bottom]])
    right_line_merged = np.array([[right_x_top, right_y_top, right_x_bottom, right_y_bottom]])
    lines_merged = np.array([left_line_merged, right_line_merged])
    return lines_merged

def draw_lines(img, lines, slope_thd=0.5, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    try:
        merged_lines = merge_lane_lines(lines, image_height=img.shape[0], slope_thd=slope_thd)
    except (ZeroDivisionError, OverflowError) as e:
        print("Error merging lines. Using raw lines instead!")
        merged_lines = lines
    
    for line in merged_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return merged_lines

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, slope_thd):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    merged_lines = draw_lines(line_img, lines, slope_thd=slope_thd, thickness=3)
    return line_img, merged_lines

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def simple_line_finding(image):
    # Tuning Parameters
    gauss_kernel_size = 5
    canny_low_thd = 50
    canny_high_thd = 180
    v_x0 = 0.1
    v_x1a = 0.472
    v_x1b = 0.528
    v_y1 = 0.60
    v_x2 = 0.94
    hough_rho = 1
    hough_theta = 2.8/180.
    hough_thd = 15
    hough_min_len = 30
    hough_max_gap = 30
    slope_thd = 0.45

    # Get Grayscale Image
    img_gray = grayscale(img=image)
    
    # Replace Yellow pixels with white in gray scale image 
    mask_yellow = cv2.inRange(image, (128, 128, 0), (255, 255, 90))
    img_gray = cv2.bitwise_or(img_gray, mask_yellow)

    # Use Gaussian Blur to filter image and remove noise
    blur_gray = gaussian_blur(img=img_gray, kernel_size=gauss_kernel_size)

    # Get Edges using "CANNY EDGE DETECTION"
    edges = canny(img=blur_gray, low_threshold=canny_low_thd, high_threshold=canny_high_thd)

    # Mask Image Using Polygon
    h = edges.shape[0]
    w = edges.shape[1]
    vertices = np.array([[(v_x0*w, h), (v_x1a*w, v_y1*h), 
                          (v_x1b*w, v_y1*h), (v_x2*w, h)]], 
                        dtype=np.int32)
    masked_edges = region_of_interest(img=edges, vertices=vertices)

    # Get Raw Lane Lines using "HOUGH LINES"
    lines_image, lines = hough_lines(img=masked_edges, rho=hough_rho, theta=hough_theta, threshold=hough_thd,
                                     min_line_len=hough_min_len, max_line_gap=hough_max_gap, slope_thd=slope_thd)

    # Overlay Images
    combined_img = weighted_img(img=lines_image, initial_img=image, α=1.0, β=0.9)
    return combined_img, lines
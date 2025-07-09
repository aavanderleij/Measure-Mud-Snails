import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, disk, h_minima
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import measure, color, morphology
from openpyxl import Workbook

def get_all_files(dir_name):
    """
    Recursively get all files in a directory.

    Parameters:
    - dir_name: str, the path to the directory to search.

    Returns:
    - file_list: list of str, a list containing the full paths of all files in the directory.
    """
    #TODO find a more efficient way to get all files
    file_list = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_file_list(folder_name, file_check):
    """
    Returns a list of files with the given extension and their count.
    """
    file_list = get_all_files(folder_name)
    file_list = [f for f in file_list if f.lower().endswith('.' + file_check.lower())]
    image_number = len(file_list)
    return image_number, file_list

def get_file_name(file_list, image_number):
    """
    Returns a list of file names without extension from a list of file paths.
    """
    file_names = []
    for i in range(image_number):
        base = os.path.basename(file_list[i])
        name = os.path.splitext(base)[0]
        file_names.append(name)
    return file_names

def hsv_filter(image_clip, hue_min, hue_max):
    """
    Converts an RGB image to HSV, applies a hue threshold, and returns the color-filtered image.

    Parameters:
    - image_clip: np.ndarray, input RGB image
    - hue_min: float, minimum hue threshold (0 to 1)
    - hue_max: float, maximum hue threshold (0 to 1)

    Returns:
    - image_color_filtered: np.ndarray, filtered image with background removed
    """
    # Convert RGB to HSV
    hsv_image = cv2.cvtColor(image_clip, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[:, :, 0] /= 179.0  # Normalize hue to [0, 1]

    # Define thresholds
    channel1_min = hue_min
    channel1_max = hue_max
    channel2_min = 0.0
    channel2_max = 1.0
    channel3_min = 0.0
    channel3_max = 1.0

    # Create mask
    mask = (
        (hsv_image[:, :, 0] >= channel1_min) & (hsv_image[:, :, 0] <= channel1_max) &
        (hsv_image[:, :, 1] >= channel2_min) & (hsv_image[:, :, 1] <= channel2_max) &
        (hsv_image[:, :, 2] >= channel3_min) & (hsv_image[:, :, 2] <= channel3_max)
    )

    # Apply mask
    image_color_filtered = np.copy(image_clip)
    image_color_filtered[~mask] = 0

    return image_color_filtered

def create_new_folder(folder_name):
    """
    Creates a new folder if it doesn't exist.

    Parameters:
    - folder_name: str, the name of the folder to create.

    Returns:
    - folder_name: str, the path to the created or existing folder.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def excel_output(data, filename, sheet_name='Sheet1'):
    """
    Outputs data to an Excel file.

    Parameters:
    - data: list of lists, the data to write to the Excel file.
    - filename: str, the name of the Excel file.
    - sheet_name: str, the name of the sheet in the Excel file.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    for row in data:
        ws.append(row)
    wb.save(filename)

def petri_dish_clipping(image_rgb, image_gray, radius_range, radius_real, radius_reduction, file_name):
    """
    Fits a circle to the petri dish and creates a mask to clip the RGB image.
    Returns the clipped RGB image and the rescale factor.
    """
    # 1. Rescale for speed
    resize_factor = 0.2
    radius_low = radius_range[0][0] * resize_factor
    radius_high = radius_range[0][1] * resize_factor
    image_gray_resize = cv2.resize(image_gray, (0, 0), fx=resize_factor, fy=resize_factor)

    # 2. Fit circles using HoughCircles
    circles = cv2.HoughCircles(
        image_gray_resize,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=int(radius_low),
        maxRadius=int(radius_high)
    )

    if circles is None or len(circles[0]) == 0:
        raise RuntimeError("No circles found in image.")

    # Take the first (strongest) circle
    circles = np.uint16(np.around(circles))
    center = circles[0][0][:2] / resize_factor  # (x, y) in original scale
    radius = circles[0][0][2] / resize_factor - radius_reduction

    # 2.2 Show control image
    fig, ax = plt.subplots()
    ax.imshow(image_gray, cmap='gray')
    circ = plt.Circle((center[0], center[1]), radius, color='r', fill=False, linewidth=2)
    ax.add_patch(circ)
    plt.title(file_name)
    plt.show()

    # 3. Create mask for clipping
    mask = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 1, thickness=-1)
    mask = mask.astype(bool)

    # 4. Calculate rescale factor
    rescale_factor = radius_real / (radius + radius_reduction)

    # 5. Apply mask to all channels
    image_clip_rgb = np.zeros_like(image_rgb)
    for l in range(3):
        channel = image_rgb[:, :, l]
        channel_masked = np.zeros_like(channel)
        channel_masked[mask] = channel[mask]
        image_clip_rgb[:, :, l] = channel_masked

    return image_clip_rgb, rescale_factor

def watershed_application(bw):
    """
    Watershed segmentation as in WatershedApplication.m.
    """
    from skimage.morphology import remove_small_holes, remove_small_objects
    from skimage.feature import peak_local_max

    # Remove small holes
    bw2 = remove_small_objects(~bw, 10)
    D = ndi.distance_transform_edt(bw)
    D = -D
    # Find local maxima
    local_maxi = peak_local_max(D, indices=False, footprint=np.ones((3, 3)), labels=bw)
    markers = ndi.label(local_maxi)[0]
    labels_ws = watershed(D, markers, mask=bw)
    # Remove watershed lines
    bw3 = bw.copy()
    bw3[labels_ws == 0] = 0
    return bw3

def region_filter_setup1(image_colour_filtered, image_data_rgb, min_area_mm, max_area_mm,
                         min_extent, max_extent, min_ecc, max_ecc, minmax_area_ratio,
                         scale_factor, new_folder_name, file_name, file_name_extension):
    """
    Morphological filter and snail detection (RegionFilterSetup1).
    Returns OutputTable (list of dicts), SnailStatistics (list), OutputImage (np.ndarray).
    """
    # 1. Transform mm to px
    min_area = min_area_mm / (scale_factor ** 2)
    max_area = max_area_mm / (scale_factor ** 2)

    # 2. Morphological operations
    bw_perimeter = morphology.binary_dilation(
        color.rgb2gray(image_colour_filtered) > 0) ^ (color.rgb2gray(image_colour_filtered) > 0)
    bw_filled = ndi.binary_fill_holes(bw_perimeter)
    selem = disk(5)
    bw_erode = morphology.erosion(bw_filled, selem)
    bw_dilate = morphology.dilation(bw_erode, selem)

    # 2a. Watershed
    bw_ws = watershed_application(bw_dilate)

    # 3. Area, extent, eccentricity filtering
    label_img = measure.label(bw_ws)
    props = measure.regionprops(label_img)
    filtered_labels = []
    max_area_found = max([p.area for p in props]) if props else 0

    for p in props:
        if (min_area <= p.area <= max_area and
            min_extent <= p.extent <= max_extent and
            min_ecc <= p.eccentricity <= max_ecc and
            p.area > max_area_found * minmax_area_ratio):
            filtered_labels.append(p.label)

    geo_filtered = np.isin(label_img, filtered_labels)
    num_snails = len(filtered_labels)

    # OutputTable and statistics
    output_table = []
    snail_stats = [0]*7
    if num_snails == 0:
        print(f"No Snails detected on File: {file_name}")
        output_image = image_data_rgb.copy()
    else:
        snail_props = measure.regionprops(measure.label(geo_filtered), intensity_image=color.rgb2gray(image_data_rgb))
        for i, p in enumerate(snail_props, 1):
            output_table.append({
                'SnailNumber': i,
                'Area': p.area,
                'Extent': p.extent,
                'MajorAxisLength': p.major_axis_length,
                'MinorAxisLength': p.minor_axis_length,
                'Perimeter': p.perimeter,
                'Centroid': p.centroid
            })
        snail_stats[0] = num_snails
        snail_stats[1] = np.median([p.area for p in snail_props])
        snail_stats[2] = np.mean([p.area for p in snail_props])
        snail_stats[3] = np.mean([p.major_axis_length for p in snail_props])
        snail_stats[4] = np.mean([p.minor_axis_length for p in snail_props])
        snail_stats[5] = np.median([p.major_axis_length for p in snail_props])
        snail_stats[6] = np.median([p.minor_axis_length for p in snail_props])

        # Output image: snails black, perimeter red, axes
        output_image = image_data_rgb.copy()
        snail_outline = morphology.binary_dilation(geo_filtered) ^ geo_filtered
        snail_filled = ndi.binary_fill_holes(snail_outline)
        output_image[snail_filled, 0] = 0
        output_image[snail_filled, 1] = 0
        output_image[snail_filled, 2] = 0
        output_image[snail_outline, 0] = 255
        output_image[snail_outline, 1] = 0
        output_image[snail_outline, 2] = 0

        # Numbering and axes
        fig, ax = plt.subplots()
        ax.imshow(output_image)
        for i, p in enumerate(snail_props, 1):
            y, x = p.centroid
            ax.text(x, y, str(i), color='white', fontsize=12, ha='center', va='center')
            # Major axis (red)
            angle = p.orientation
            x0 = x + np.cos(angle) * 0.5 * p.major_axis_length
            y0 = y - np.sin(angle) * 0.5 * p.major_axis_length
            x1 = x - np.cos(angle) * 0.5 * p.major_axis_length
            y1 = y + np.sin(angle) * 0.5 * p.major_axis_length
            ax.plot([x0, x1], [y0, y1], '-', color='red', linewidth=1.2)
            # Minor axis (yellow)
            angle2 = angle + np.pi/2
            x0 = x + np.cos(angle2) * 0.5 * p.minor_axis_length
            y0 = y - np.sin(angle2) * 0.5 * p.minor_axis_length
            x1 = x - np.cos(angle2) * 0.5 * p.minor_axis_length
            y1 = y + np.sin(angle2) * 0.5 * p.minor_axis_length
            ax.plot([x0, x1], [y0, y1], '-', color='yellow', linewidth=1.2)
        plt.title(file_name)
        plt.axis('off')
        plt.savefig(os.path.join(new_folder_name, f"{file_name}{file_name_extension}"), dpi=300)
        plt.close(fig)

    return output_table, snail_stats, output_image

def region_filter_setup2(image_colour_filtered, image_data_rgb, min_area_mm, max_area_mm,
                         min_extent, max_extent, min_ecc, max_ecc, minmax_area_ratio,
                         scale_factor, new_folder_name, file_name, file_name_extension):
    """
    Morphological filter and snail detection (RegionFilterSetup2).
    Returns OutputTable (list of dicts), SnailStatistics (list), OutputImage (np.ndarray).
    """
    # 1. Transform mm to px
    min_area = min_area_mm / (scale_factor ** 2)
    max_area = max_area_mm / (scale_factor ** 2)

    # 2. Morphological operations
    bw_perimeter = morphology.binary_dilation(
        color.rgb2gray(image_colour_filtered) > 0) ^ (color.rgb2gray(image_colour_filtered) > 0)
    bw_filled = ndi.binary_fill_holes(bw_perimeter)
    selem = disk(5)
    bw_erode = morphology.erosion(bw_filled, selem)
    bw_dilate = morphology.dilation(bw_erode, selem)

    # 3. Area, extent, eccentricity filtering
    label_img = measure.label(bw_dilate)
    props = measure.regionprops(label_img)
    filtered_labels = []
    max_area_found = max([p.area for p in props]) if props else 0

    for p in props:
        if (min_area <= p.area <= max_area and
            min_extent <= p.extent <= max_extent and
            min_ecc <= p.eccentricity <= max_ecc and
            p.area > max_area_found * minmax_area_ratio):
            filtered_labels.append(p.label)

    geo_filtered = np.isin(label_img, filtered_labels)
    num_snails = len(filtered_labels)

    # OutputTable and statistics
    output_table = []
    snail_stats = [0]*7
    if num_snails == 0:
        print(f"No Snails detected on File: {file_name}")
        output_image = image_data_rgb.copy()
    else:
        snail_props = measure.regionprops(measure.label(geo_filtered), intensity_image=color.rgb2gray(image_data_rgb))
        for i, p in enumerate(snail_props, 1):
            output_table.append({
                'SnailNumber': i,
                'Area': p.area,
                'Extent': p.extent,
                'MajorAxisLength': p.major_axis_length,
                'MinorAxisLength': p.minor_axis_length,
                'Perimeter': p.perimeter,
                'Centroid': p.centroid
            })
        snail_stats[0] = num_snails
        snail_stats[1] = np.median([p.area for p in snail_props])
        snail_stats[2] = np.mean([p.area for p in snail_props])
        snail_stats[3] = np.mean([p.major_axis_length for p in snail_props])
        snail_stats[4] = np.mean([p.minor_axis_length for p in snail_props])
        snail_stats[5] = np.median([p.major_axis_length for p in snail_props])
        snail_stats[6] = np.median([p.minor_axis_length for p in snail_props])

        # Output image: snails black, perimeter red, axes
        output_image = image_data_rgb.copy()
        snail_outline = morphology.binary_dilation(geo_filtered) ^ geo_filtered
        snail_filled = ndi.binary_fill_holes(snail_outline)
        output_image[snail_filled, 0] = 0
        output_image[snail_filled, 1] = 0
        output_image[snail_filled, 2] = 0
        output_image[snail_outline, 0] = 255
        output_image[snail_outline, 1] = 0
        output_image[snail_outline, 2] = 0

        # Numbering and axes
        fig, ax = plt.subplots()
        ax.imshow(output_image)
        for i, p in enumerate(snail_props, 1):
            y, x = p.centroid
            ax.text(x, y, str(i), color='white', fontsize=12, ha='center', va='center')
            # Major axis (red)
            angle = p.orientation
            x0 = x + np.cos(angle) * 0.5 * p.major_axis_length
            y0 = y - np.sin(angle) * 0.5 * p.major_axis_length
            x1 = x - np.cos(angle) * 0.5 * p.major_axis_length
            y1 = y + np.sin(angle) * 0.5 * p.major_axis_length
            ax.plot([x0, x1], [y0, y1], '-', color='red', linewidth=1.2)
            # Minor axis (yellow)
            angle2 = angle + np.pi/2
            x0 = x + np.cos(angle2) * 0.5 * p.minor_axis_length
            y0 = y - np.sin(angle2) * 0.5 * p.minor_axis_length
            x1 = x - np.cos(angle2) * 0.5 * p.minor_axis_length
            y1 = y + np.sin(angle2) * 0.5 * p.minor_axis_length
            ax.plot([x0, x1], [y0, y1], '-', color='yellow', linewidth=1.2)
        plt.title(file_name)
        plt.axis('off')
        plt.savefig(os.path.join(new_folder_name, f"{file_name}{file_name_extension}"), dpi=300)
        plt.close(fig)

    return output_table, snail_stats, output_image

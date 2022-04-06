import re
import numpy as np

from timeit import default_timer as timer

import color_conversion
import histogram
import noise
import filter
import edge_detection
import morphological_filters
import segmentation


total_time = {}
img_count = 0


def run_functions(img, functions, img_name):
    global img_count
    img_count += 1

    for func in functions:
        img = execute_function(func, img, img_name)

    return img


def execute_function(func, img, img_name):
    if func[0] == 'color_conversion':
        start_time = timer()

        color_img = color_conversion.convert_single_spectrum(img, func[1])

        add_time(timer() - start_time, 'Color Conversion')

        return color_img
    elif func[0] == 'histogram':
        if func[1] == 'generate':
            start_time = timer()

            img_hist = histogram.generate_histogram(img)
            histogram.save_histogram(img_hist, img_name, 'hist')

            add_time(timer() - start_time, 'Histogram Generation')
        elif func[1] == 'average':
            img_class = re.sub(r'\d+', '', img_name)

            start_time = timer()

            img_hist = histogram.average_histogram(img, img_class)
            histogram.save_histogram(img_hist, img_class, 'class_hist')

            add_time(timer() - start_time, 'Average Histogram')
        elif func[1] == 'equalize':
            start_time = timer()

            equ_img = histogram.equalize_histogram(img)
            img_hist = histogram.generate_histogram(equ_img)
            histogram.save_histogram(img_hist, img_name, 'equ_hist')

            add_time(timer() - start_time, 'Histogram Equalization')

            return equ_img
        elif func[1] == 'quantize':
            start_time = timer()

            quan_img, msqe = histogram.quantize_histogram(img, int(func[2]))
            img_hist = histogram.generate_histogram(quan_img)
            histogram.save_histogram(img_hist, img_name, 'quan_hist')

            add_time(timer() - start_time, 'Histogram Quantization')

            print('MSQE of ' + img_name + ': ' + str(msqe.round(3)))

            return quan_img
    elif func[0] == 'noise':
        if func[1] == 'salt_pepper':
            start_time = timer()

            sp_img = noise.add_salt_pepper_noise(img, float(func[2]))

            add_time(timer() - start_time, 'Salt & Pepper Noise')

            return sp_img
        elif func[1] == 'gaussian':
            start_time = timer()

            gaus_img = noise.add_gaussian_noise(img, float(func[2]), float(func[3]))

            add_time(timer() - start_time, 'Gaussian Noise')

            return gaus_img
    elif func[0] == 'filter':
        weights = convert_to_array(func[3], int(func[2]))

        if func[1] == 'linear':
            start_time = timer()

            lin_img = filter.apply_linear_filter(img, int(func[2]), weights)

            add_time(timer() - start_time, 'Linear Filter')

            return lin_img
        elif func[1] == 'median':
            start_time = timer()

            med_img = filter.apply_median_filter(img, int(func[2]), weights)

            add_time(timer() - start_time, 'Median Filter')

            return med_img
    elif func[0] == 'edge':
        if func[1] == 'prewitt':
            start_time = timer()

            edge_img = edge_detection.prewitt_operator(img)

            add_time(timer() - start_time, 'Prewitt Edge Operator')

            return edge_img
        elif func[1] == 'sobel':
            start_time = timer()

            edge_img = edge_detection.sobel_operator(img)

            add_time(timer() - start_time, 'Sobel Edge Operator')

            return edge_img
    elif func[0] == 'morph':
        if func[1] == 'dilation':
            start_time = timer()

            dil_img = morphological_filters.dilation(img, int(func[2]))

            add_time(timer() - start_time, 'Dilation')

            return dil_img
        elif func[1] == 'erosion':
            start_time = timer()

            ero_img = morphological_filters.erosion(img, int(func[2]))

            add_time(timer() - start_time, 'Erosion')

            return ero_img
    elif func[0] == 'segmentation':
        if func[1] == 'histo_thresholding':
            pass
        elif func[1] == 'clustering':
            pass

    return img


def convert_to_array(arr_string, size):
    array = re.findall(r'(\d+(?:\.\d+)?)', arr_string)
    return np.array(array).reshape(size, size).astype(np.float32)


def add_time(time, func):
    if func not in total_time.keys():
        total_time[func] = time
    else:
        total_time[func] += time


def display_stats():
    print('\n------------- Stats -------------')
    print('Total batch time for each process:')

    for process, time in total_time.items():
        formatted_time = round(time, 3)
        print('\t' + process + ': ' + str(formatted_time) + ' s')

    print('\nAverage time per image for each process:')

    for process, time in total_time.items():
        avg_time = time / img_count
        formatted_time = round(avg_time, 3)
        print('\t' + process + ': ' + str(formatted_time) + ' s')

    print('--------------------------------------')

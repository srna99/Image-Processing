import re
import numpy as np

import color_conversion
import histogram
import noise
import filter


def execute_function(func, img, img_name):
    print(func)

    if func[0] == 'color_conversion':
        return color_conversion.convert_single_spectrum(img, func[1])
    elif func[0] == 'histogram':
        if func[1] == 'generate':
            img_hist = histogram.generate_histogram(img)
            histogram.save_histogram(img_hist, img_name, 'hist')
        elif func[1] == 'average':
            img_class = re.sub(r'\d+', '', img_name)

            img_hist = histogram.average_histogram(img, img_class)
            histogram.save_histogram(img_hist, img_class, 'class_hist')
        elif func[1] == 'equalize':
            return histogram.equalize_histogram(img)
        elif func[1] == 'quantize':
            return histogram.quantize_histogram(img, int(func[2]))
    elif func[0] == 'noise':
        if func[1] == 'salt_pepper':
            return noise.add_salt_pepper_noise(img, float(func[2]))
        elif func[1] == 'gaussian':
            return noise.add_gaussian_noise(img, float(func[2]), float(func[3]))
    elif func[0] == 'filter':
        weights = convert_to_array(func[3], int(func[2]))

        if func[1] == 'linear':
            return filter.apply_linear_filter(img, int(func[2]), weights)
        elif func[1] == 'median':
            return filter.apply_median_filter(img, int(func[2]), weights)

    return img


def convert_to_array(arr_string, size):
    array = re.findall(r'(\d+(?:\.\d+)?)', arr_string)
    return np.array(array).reshape(size, size).astype(np.float32)

import re

import color_conversion
import histogram
import noise


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
            equ_img = histogram.equalize_histogram(img)
            img_hist = histogram.generate_histogram(equ_img)
            histogram.save_histogram(img_hist, img_name, 'equ_hist')

            return equ_img
        elif func[1] == 'quantize':
            quan_img = histogram.quantize_histogram(img, int(func[2]))
            img_hist = histogram.generate_histogram(quan_img)
            histogram.save_histogram(img_hist, img_name, 'quan_hist')

            return quan_img
    elif func[0] == 'noise':
        if func[1] == 'salt_pepper':
            return noise.add_salt_pepper_noise(img, float(func[2]))
        elif func[1] == 'gaussian':
            return noise.add_gaussian_noise(img, float(func[2]), float(func[3]))

    return img

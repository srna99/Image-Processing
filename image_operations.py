import re

import color_conversion
import histogram


def execute_function(func, img, img_name):
    if func[0] == 'color_conversion':
        return color_conversion.convert_single_spectrum(img, func[1])
    elif func[0] == 'hist':
        img_class = re.sub(r'\d+', '', img_name)

        if func[1] == 'generate':
            img_hist = histogram.generate_histogram(img)
            histogram.save_histogram(img_hist, img_name, 'hist')
        elif func[1] == 'average':
            img_hist = histogram.average_histogram(img, img_class)
            histogram.save_histogram(img_hist, img_class, 'avg_hist')
        elif func[1] == 'equalize':
            img_hist = histogram.equalize_histogram(img)
            histogram.save_histogram(img_hist, img_name, 'equ_hist')

    return img

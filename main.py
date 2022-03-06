import glob
import os
import sys
import numpy as np

from pathlib import Path
from PIL import Image

import color_conversion


def clean_output():
    if not os.path.isdir('output'):
        os.makedirs('output')
    else:
        for file in glob.glob('output/*.BMP'):
            try:
                os.remove(file)
            except OSError as e:
                print("Error - ", file, ":", e)


def execute_function(func, img):
    print(func)

    if func[0] == 'color_conversion':
        return color_conversion.convert_single_spectrum(img, func[1])


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        filepath = f.readline().strip()
        functions = [line.strip().split() for line in f.readlines()]
        # print(filepath)
        # print(functions)

    clean_output()

    img_count = 0
    for file in glob.glob(os.path.join(filepath, '*.BMP')):
        try:
            filename = Path(file).stem
            print(filename)

            with Image.open(file) as ori_img:
                ori_img.show()
                copy_img = np.array(ori_img)
        except IOError as e:
            print("Error - ", filepath, ":", e)
            continue

        for func in functions:
            copy_img = execute_function(func, copy_img)

        mod_img = Image.fromarray(copy_img)
        mod_img.save(os.path.join('output', filename + '_mod.BMP'))
        mod_img.show()

        img_count += 1
        if img_count > 0:
            break

    # print(img_count)

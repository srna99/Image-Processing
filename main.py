import glob
import os
import sys
import numpy as np

from pathlib import Path
from PIL import Image

import image_operations
import feature_extraction


def clean_output():
    if not os.path.isdir('output'):
        os.makedirs('output')
    else:
        for file in glob.glob(os.path.join('output', '*')):
            try:
                os.remove(file)
            except OSError as e:
                print('Error - ', file, ':', e)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        filepath = f.readline().strip()
        functions = [line.strip().split() for line in f.readlines()]

    clean_output()

    for file in glob.glob(os.path.join(filepath, '*.BMP')):
        try:
            filename = Path(file).stem

            with Image.open(file) as ori_img:
                gray_img = np.round(np.mean(np.array(ori_img.copy()), axis=2))
        except IOError as e:
            print('Error - ', filepath, ':', e)
            continue

        copy_img = np.uint8(gray_img)

        if functions == [['features']]:
            image_operations.run_functions(copy_img, functions, filename)
        else:
            copy_img = image_operations.run_functions(copy_img, functions, filename)

            mod_img = Image.fromarray(copy_img)
            mod_img.save(os.path.join('output', filename + '_mod.BMP'))

    if functions == [['features']]:
        feature_extraction.save_training_set()

    image_operations.display_stats()

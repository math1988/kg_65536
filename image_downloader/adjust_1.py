import os
from joblib import Parallel, delayed
import itertools
import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image

# Step 1: Read unalt file structure
list_paths = []
export_base_folder = '/home/mathematics/kaggle/furniture/data/validation/'
pic_size = 299


for subdir, dirs, files in os.walk("/mnt/BC42459D42455CF0/FGVC/validation/"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        list_paths.append(filepath)

# Please modify based on the file structure

def gen_alt(path, export_base_folder = None):
    pic = Image.open(path)
    pic = pic.resize((pic_size,pic_size),Image.ANTIALIAS)
    class_string = path.split('/')[-1].split('.')[0].split('_')[-1]
    folder = export_base_folder + '/' + str(pic_size) + '/'+ str(int(class_string)-1) + '/'
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except:
            print("ERROR")

    pic.save(folder + '/' +
             path.split('/')[-1] +
             '.jpeg', 'JPEG', quality=100)

Parallel(n_jobs=16, verbose = 10)(delayed(gen_alt)(path,

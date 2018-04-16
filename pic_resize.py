import sys, os
from joblib import Parallel, delayed
import itertools
import numpy as np

from flag_parser import parse_flag

try:
    import Image
except ImportError:
    from PIL import Image

def gen_alt(path, export_base_folder, pic_size):
    pic = Image.open(path)
    pic = pic.resize((pic_size,pic_size),Image.ANTIALIAS)

    # |path| is like "<directory>/<image_id>_<label_id>.jpeg" 
    class_string = path.split('/')[-1].split('.')[0].split('_')[-1]

    folder = "{0}/Size_{1}/Class_{2}/".format(export_base_folder, str(pic_size), str(int(class_string)-1))
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except:
            print("Error Making Folder: {0}".format(folder))

    pic.save(folder + os.sep +
             path.split(os.sep)[-1] +
             '.jpeg', 'JPEG', quality=100)

def Run():
    try:
        flag_map = parse_flag(sys.argv)
        import_base_folder = flag_map["input_dir"]
        export_base_folder = flag_map["output_dir"]
        pic_size = int(flag_map["image_size"])
    except:
        print('Syntax: {0}\\\n --input_dir=<Directory/>\\\n --output_dir=<Directory/>\\\n --image_size=<Number/>'
              .format(sys.argv[0]))
        sys.exit(0)
    list_paths = []

    for subdir, dirs, files in os.walk(import_base_folder):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            list_paths.append(filepath)

    Parallel(n_jobs=16, verbose = 100)(delayed(gen_alt)(path, export_base_folder, pic_size) for path in list_paths);

if __name__ == "__main__":
    Run()

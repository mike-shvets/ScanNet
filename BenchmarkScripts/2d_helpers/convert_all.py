import argparse
import multiprocessing as mp
import numpy as np
import imageio
import os
import csv
from functools import partial
import time
from datetime import timedelta
import tqdm


def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping


def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k,v in label_mapping.items():
        mapped[image==k] = v
    return mapped.astype(np.uint8)


def make_instance_image(label_image, instance_image):
    output = np.zeros_like(instance_image, dtype=np.uint16)
    # oldinst2labelinst = {}
    label_instance_counts = {}
    old_insts = np.unique(instance_image)
    for inst in old_insts:
        label = label_image[instance_image==inst][0]
        if label in label_instance_counts:
            inst_count = label_instance_counts[label] + 1
            label_instance_counts[label] = inst_count
        else:
            inst_count = 1
            label_instance_counts[label] = inst_count
        # oldinst2labelinst[inst] = (label, inst_count)
        output[instance_image==inst] = label * 1000 + inst_count
    return output


def convert_one(label_map, input_label_file, output_label_file, input_instance_file, output_instance_file):
    label_image = np.array(imageio.imread(input_label_file))
    mapped_label = map_label_image(label_image, label_map)
    imageio.imwrite(output_label_file, mapped_label)

    instance_image = np.array(imageio.imread(input_instance_file))
    mapped_label = map_label_image(label_image, label_map)
    output_instance_image = make_instance_image(mapped_label, instance_image)
    imageio.imwrite(output_instance_file, output_instance_image)


def convert_scene(scene_name, root, output_root):
    assert os.path.exists(output_root)
    output_label_folder = os.path.join(output_root, "scans", scene_name, "label-filt-nyu40")
    output_instance_folder = os.path.join(output_root, "scans", scene_name, "instance-filt-nyu40")
    os.makedirs(output_label_folder, exist_ok=True)
    os.makedirs(output_instance_folder, exist_ok=True)

    labelmap_file= os.path.join(root, "scannetv2-labels.combined.tsv")
    assert os.path.isfile(labelmap_file), f"Label map file not found: {labelmap_file}."
    label_map = read_label_mapping(labelmap_file, label_from='id', label_to='nyu40id')
    
    label_folder = os.path.join(root, "scans", scene_name, "label-filt")
    instance_folder = os.path.join(root, "scans", scene_name, "label-filt")

    filenames = os.listdir(label_folder)
    start_time = time.time()
    print(f"Converting {len(filenames)} files for {scene_name}...")
    # for fname in tqdm.tqdm(filenames):
    for fname in filenames:
        input_label_file = os.path.join(label_folder, fname)
        input_instance_file = os.path.join(instance_folder, fname)
        output_label_file = os.path.join(output_label_folder, fname)
        output_instance_file = os.path.join(output_instance_folder, fname)
        convert_one(label_map, input_label_file, output_label_file, input_instance_file, output_instance_file)
    time_elapsed = time.time() - start_time
    time_elapsed = str(timedelta(seconds=time_elapsed))
    print(f"Done processing {scene_name} in {time_elapsed}.")


def main():
    parser = argparse.ArgumentParser("Prepare NYU40 annotation for ScanNet")
    parser.add_argument(
        "scannet_root",
        help="Path to scannet root. The folder must contain scans subfolder and label map scannetv2-labels.combined.tsv"
    )
    parser.add_argument(
        "output_root",
        help="Path to the output root."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes."
    )
    args = parser.parse_args()

    root = args.scannet_root
    scans_folder = os.path.join(root, "scans")

    assert os.path.isdir(scans_folder), f"Scans folder does not exist: {scans_folder}."
    scene_names = sorted(os.listdir(scans_folder))

    num_processes = min(args.num_processes, mp.cpu_count())
    num_processes = min(num_processes, len(scene_names))
    if num_processes > 1:
        print(f"Setting up a pool of {num_processes} processes.")
        fn = partial(convert_scene, root=root, output_root=args.output_root)
        pool = mp.Pool(num_processes)
        pool.map(fn, scene_names)
        pool.close()
    else:
        for scene_name in scene_names:
            convert_scene(scene_name, root, args.output_root)


if __name__ == "__main__":
    main()

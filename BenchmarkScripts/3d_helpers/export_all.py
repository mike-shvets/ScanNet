import argparse
import multiprocessing as mp
import os
from functools import partial

from export_train_mesh_for_evaluation import export


def export_one(scene_name, root, output_root):
    mesh_file = os.path.join(root, "scans", scene_name, scene_name + '_vh_clean_2.ply')
    agg_file = os.path.join(root, "scans", scene_name, scene_name + '.aggregation.json')
    seg_file = os.path.join(root, "scans", scene_name, scene_name + '_vh_clean_2.0.010000.segs.json')
    label_map_file = os.path.join(root, "scannetv2-labels.combined.tsv")

    print(f"Exporting gt for {scene_name}...")

    os.makedirs(os.path.join(output_root, "3d_label_gt"), exist_ok=True)
    # output_file = os.path.join(output_root, "scans", scene_name, f"{scene_name}_3d_label_gt.txt")
    output_file = os.path.join(output_root, "3d_label_gt", f"{scene_name}.txt")
    export(mesh_file, agg_file, seg_file, label_map_file, "label", output_file)

    # output_file = os.path.join(output_root, "scans", scene_name, f"{scene_name}_3d_instance_gt.txt")
    # export(mesh_file, agg_file, seg_file, label_map_file, "instance", output_file)


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

    assert os.path.exists(args.output_root)

    assert os.path.isdir(scans_folder), f"Scans folder does not exist: {scans_folder}."
    scene_names = sorted(os.listdir(scans_folder))

    num_processes = min(args.num_processes, mp.cpu_count())
    num_processes = min(num_processes, len(scene_names))
    if num_processes > 1:
        print(f"Setting up a pool of {num_processes} processes.")
        fn = partial(export_one, root=root, output_root=args.output_root)
        pool = mp.Pool(num_processes)
        pool.map(fn, scene_names)
        pool.close()
    else:
        for scene_name in scene_names:
            export_one(scene_name, root, args.output_root)


if __name__ == "__main__":
    main()

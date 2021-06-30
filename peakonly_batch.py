#!python
# A batch interface for peakonly

# This is a pretty ugly hack, essentially a simplified reimlementation of
# processing_utils.runner.FilesRunner._batch_run
# At the time of writing it gives identical results to the Peakonly GUI.

# Chris MacRaild, Monash University, 2021

import os
import sys
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch

from processing_utils.postprocess import ResultTable
from processing_utils.roi import get_ROIs
from processing_utils.matching import construct_mzregions, rt_grouping, align_component
from processing_utils.run_utils import preprocess, get_borders
from processing_utils.run_utils import border_correction, build_features, feature_collapsing
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator


def build_parser():
    parser = argparse.ArgumentParser(description="""Run Peakonly from the commandline,
                                     bypassing the gui.""")
    parser.add_argument('--mzML', metavar='mzMLfiles', nargs='*', action='append', help='Files to process')
    parser.add_argument('--mz-deviation', help='m/z tolerance for defining ROIs',
                        type=float, default=0.005)
    parser.add_argument('--min-ROI-length', help='Minimum length of ROIs, in data points',
                        type=int, default=15)
    parser.add_argument('--max-zeros', help='Maximum no. of consecutive zero points to allow in ROIs',
                        type=int, default=3)
    parser.add_argument('--min-peak-length', help='Minimum length of peaks, in data points',
                        type=int, default=8)
    parser.add_argument('--model-dir', metavar='directory', help="""Load model weights
                        from directory""", type=str, default="./data/weights/")
    parser.add_argument('--csv', metavar='csvFiles', help="""Write feature table to file""",
                        type=str, default='peakonly.csv')
    parser.add_argument('--png', metavar='directory', help="""Plot features as png files to directory""",
                        type=str)
    return parser


if __name__ == "__main__":

    arg_parser = build_parser()
    args = arg_parser.parse_args()

    files = []
    if args.mzML is not None:
        for tok in args.mzML:
            for name in glob.glob(tok):
                files.append(name)
    if not files:
        arg_parser.error("No mzML files specified. Nothing to do.")

    if not (os.path.exists(os.path.join(args.model_dir, "Classifier.pt")) and
            os.path.exists(os.path.join(args.model_dir, "Segmentator.pt"))):
        arg_parser.error(f"""Can't find model weights at {args.model_dir}
                         Use the Peakonly GUI to download the models, and
                         specify their location with --model-dir.""")

    print("Finding ROIs...")
    rois = {}
    for f in files:
        rois[f] = get_ROIs(f, args.mz_deviation, args.min_ROI_length, args.max_zeros, None)
    mzregions = construct_mzregions(rois, args.mz_deviation)
    components = rt_grouping(mzregions)

    print("Aligning ROIs...")
    aligned_components = []
    for i, component in enumerate(components):
        aligned_components.append(align_component(component))

    print("Finding peaks...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier().to(device)
    path2classifier_weights = os.path.join(args.model_dir, "Classifier.pt")
    classifier.load_state_dict(torch.load(path2classifier_weights, map_location=device))
    classifier.eval()
    segmentator = Segmentator().to(device)
    path2segmentator_weights = os.path.join(args.model_dir, "Segmentator.pt")
    segmentator.load_state_dict(torch.load(path2segmentator_weights, map_location=device))
    segmentator.eval()

    component_number = 0
    features = []
    for j, component in enumerate(aligned_components):  # run through components
        borders = {}
        to_delete = []
        for i, (sample, roi) in enumerate(zip(component.samples, component.rois)):
            signal = preprocess(roi.i, device, interpolate=True, length=256)
            classifier_output, _ = classifier(signal)
            classifier_output = classifier_output.data.cpu().numpy()
            label = np.argmax(classifier_output)
            if label == 1:
                _, segmentator_output = segmentator(signal)
                segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()
                borders[sample] = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :],
                                              peak_minimum_points=args.min_peak_length,
                                              interpolation_factor=len(signal[0, 0]) / len(roi.i))
            else:
                to_delete.append(i)
        if len(borders) > len(files) // 3:  # enough rois contain a peak
            component.pop(to_delete)  # delete ROIs which don't contain peaks
            border_correction(component, borders)
            features.extend(build_features(component, borders, component_number))
            component_number += 1

    features = feature_collapsing(features)
    to_delete = []
    for i, feature in enumerate(features):
        if len(feature) <= len(files) // 3:  # to do: adjustable parameter
            to_delete.append(i)
    for j in to_delete[::-1]:
        features.pop(j)
    print('total number of features: {}'.format(len(features)))
    features.sort(key=lambda x: x.mz)

    if args.csv:
        table = ResultTable(files, features)
        table.fill_zeros(args.mz_deviation)
        print("Writing output...")
        table.to_csv(args.csv)

    if args.png:
        fig = plt.figure()
        for i, feature in enumerate(features):
            ax = fig.add_subplot(111)
            feature.plot(ax, shifted=True)
            try:
                fig.savefig(os.path.join(args.png, f'{i}.png'))
            except FileNotFoundError:
                os.mkdir(args.png)
                fig.savefig(os.path.join(args.png, f'{i}.png'))
            fig.clear()
        plt.close(fig)

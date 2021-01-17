import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


class Preprocessor:
    def __init__(self, input, output, type):
        self.input = input
        self.output = output
        self.type = type
        self.filenames = os.listdir(self.input)
        try:
            self.filenames.remove("Masks")
        except ValueError:
            pass
        try:
            self.filenames.remove("Edges")
        except ValueError:
            pass

    def xml2mask_and_edge(self, filename):
        mask = np.zeros([1000, 1000], dtype=np.uint8)
        edge = [[255 for i in range(0, 1000)] for i in range(0, 1000)]
        edge = np.asarray(edge, dtype=np.uint8)

        xml = str(self.input + "/" + filename + ".xml")
        tree = ET.parse(xml)
        root = tree.getroot()
        regions = root.findall("Annotation/Regions/Region")
        for region in regions:
            points = []
            for point in region.findall("Vertices/Vertex"):
                x = float(point.attrib["X"])
                y = float(point.attrib["Y"])
                points.append([x, y])

            points = np.asarray([points], dtype=np.int32)
            cv2.fillPoly(img=mask, pts=points, color=255)
            cv2.polylines(img=edge, pts=points, color=0, thickness=2, isClosed=True)
        if not os.path.exists(self.output + "/Masks"):
            os.mkdir(self.output + "/Masks")
        if not os.path.exists(self.output + "/Edges"):
            os.mkdir(self.output + "/Edges")
        cv2.imwrite(self.output + "/Masks/" + filename + ".png", mask)
        cv2.imwrite(self.output + "/Edges/" + filename + ".png", edge)

    def patch(self, filename, patch_size):
        img = cv2.imread(self.output + "/Tissue Images/" + filename + self.type)
        img = np.pad(img, ((12, 12), (12, 12), (0, 0)), "symmetric")

        mask = cv2.imread(self.output + "/Masks/" + filename + ".png")
        mask = np.pad(mask, ((12, 12), (12, 12), (0, 0)), "symmetric")

        edge = cv2.imread(self.output + "/Edges/" + filename + ".png")
        edge = np.pad(edge, ((12, 12), (12, 12), (0, 0)), "symmetric")

        num_rows = int((img.shape[0] - 256) / 64)
        num_cols = int((img.shape[1] - 256) / 64)

        if not os.path.exists(self.output + "/Patch"):
            os.mkdir(self.output + "/Patch")
        if not os.path.exists(self.output + "/Patch/Imgs"):
            os.mkdir(self.output + "/Patch/Imgs")
        if not os.path.exists(self.output + "/Patch/Masks"):
            os.mkdir(self.output + "/Patch/Masks")
        if not os.path.exists(self.output + "/Patch/Edges"):
            os.mkdir(self.output + "/Patch/Edges")

        for r in tqdm(range(num_rows)):
            for c in tqdm(range(num_cols)):
                patch_img = img[r * 64: r * 64 + patch_size, c * 64: c * 64 + patch_size, :]
                patch_mask = mask[r * 64: r * 64 + patch_size, c * 64: c * 64 + patch_size, :]
                patch_edge = edge[r * 64: r * 64 + patch_size, c * 64: c * 64 + patch_size, :]

                cv2.imwrite(self.output + "/Patch/Imgs/" + filename + str(r) + "_" + str(c) + ".png", patch_img)
                cv2.imwrite(self.output + "/Patch/Masks/" + filename + str(r) + "_" + str(c) + ".png", patch_mask)
                cv2.imwrite(self.output + "/Patch/Edges/" + filename + str(r) + "_" + str(c) + ".png", patch_edge)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="directory to annotations", default="./")
    parser.add_argument("--output", type=str, help="output directory", default="./")
    parser.add_argument("--type", type=str, help="type of images", default=".png")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    input = args.input
    output = args.output
    type = args.type

    preprocessor = Preprocessor(input, output, type)
    split_filenames = []
    for filename in preprocessor.filenames:
        split_filename, _ = os.path.splitext(filename)
        split_filenames.append(split_filename)

    for filename in tqdm(split_filenames):
        preprocessor.xml2mask_and_edge(filename)
        preprocessor.patch(filename, 256)
    print("Done!")

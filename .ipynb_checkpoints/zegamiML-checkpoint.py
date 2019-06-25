#!/usr/bin/env python3

DESC = ("zegamiML - dimensionality reduction applied on the set of images "
        "to be used in Zegami processing")

import os
import argparse
import logging

import numpy as np
import pandas as pd

MISSING_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
except ImportError:
    MISSING_MATPLOTLIB = True

from PIL import Image
from sklearn.decomposition import PCA 
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import umap

# Set defaults
Image.LOAD_TRUNCATED_IMAGES = True
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
logging_format = '%(asctime)s\t%(levelname)s\t%(message)s'

# setup a standard image size; 
# this will distort some images but will get everything into the same shape
STANDARD_SIZE = (100, 100)

def main():
    parser = argparse.ArgumentParser(
            prog = "zegamiML",
            description = DESC
    )
    parser.add_argument(
            '-i', '--input', 
            help = 'Input file name (must be a tab-delimited file)',
            required = True
    ) 
    parser.add_argument(
            '-o', '--output', 
            help = ('Output file name (original TSV with the X and Y ' 
                    'columns  appended)'),
            required = True
    )
    parser.add_argument(
            '-c', '--image_column', 
            help = 'Column name in TSV that contains the image name',
            required = True
    )
    parser.add_argument(
            '-d', '--image_dir', 
            help = 'Directory that contains all the images.',
            required = True
    )
    parser.add_argument(
            '-a', '--analysis_type', 
            help = 'PCA, TSNE or UMAP.',
            required = False, 
            default = "PCA"
    )
    parser.add_argument(
            '-v', '--verbose', 
            action = 'store_true', 
            required = False
    )
    parser.add_argument(
            '-p', '--plot', 
            help = ('Produce a visualisation of the applied dimensionality'
                    'reduction.'),
            action = 'store_true', 
            required = False
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
                level = logging.DEBUG,
                format = logging_format
        )
    else:
        logging.basicConfig(
                level = logging.INFO,
                format = logging_format
        )

    
    reduce_and_plot(
            zegami_table_in = args.input, 
            zegami_table_out = args.output, 
            image_path = args.image_dir,
            image_col = args.image_column,
            analysis_type = args.analysis_type, 
            plot = args.plot
    )

def test(analysis_type):
    logging.basicConfig(
            level = logging.DEBUG,
            format = logging_format
    )

    reduce_and_plot(
            zegami_table_in = "./test/zegami.tab",
            zegami_table_out = "zegami_ML.tab", 
            image_path = "./test/imgs",
            image_col = "path",
            analysis_type = analysis_type, 
            plot = True
    )

def reduce_and_plot(zegami_table_in, zegami_table_out, image_path, 
        image_col, analysis_type, plot):
    
    # read in Zegami metadata table
    zt = pd.read_csv(zegami_table_in, sep = "\t", header = 0)
    rows, cols = zt.shape
    if cols < 2:
        err_msg = (f"Zegami table of an invalid shape {rows} x {cols}."
                   "Make sure input is tab-deliminated.")
        logging.error(err_msg)
        raise Exception(err_msg)
        
    # get a list of images via path
    images = [os.path.join(image_path, image) for image in zt[image_col]]
    
    # process images
    data, skipped = process_images(images)
    
    # reduce dimensions
    reduced_df = reduce_dimensions(data, analysis_type = analysis_type)
    
    # create labels for columns    
    x_label = analysis_type + "_x"
    y_label = analysis_type + "_y"
    
    # merge input and reduced data frames
    reduced_df = reduced_df.rename(index = str, 
                                   columns = {'x': x_label, 
                                              'y': y_label})
    reduced_df[image_col] = [os.path.basename(image_fp) for image_fp in images
            if image_fp not in skipped]
    
    zt_out = pd.merge(zt, reduced_df, on = image_col, how = 'outer')

    # plot 
    if plot:
        if MISSING_MATPLOTLIB:
            logging.error("Matplotlib module not installed! Cannot plot.")
        else:
            plt.scatter(reduced_df[x_label], reduced_df[y_label])
            plt.show()
    
    # save output
    zt_out.to_csv(zegami_table_out, na_rep = "NaN", sep = "\t", index = False)

def process_images(images):
    data = []
    skipped = []

    for image in images:
        try:
            img = Image.open(image)
        except FileNotFoundError:
            logging.warning('File %s not found! Skipping.', image)
            skipped.append(image)
            continue

        try:
            img.load()
        except IOError:
            logging.warning('%s cannot be loaded! Skipping.', image)
            skipped.append(image)
            continue
        
        img_mat = img_to_matrix(image)
        img_mat_flat = flatten_image(img_mat)
        data.append(img_mat_flat) 

    return data, skipped

def reduce_dimensions(data, analysis_type = "PCA"):
    """
    reduces dimensionality of a given data set applying one of 3 supported
    methods: PCA, MDS, tSNE and UMAP.

    input
    """
    
    supported_types = ["PCA", "TSNE", "UMAP", "MDS"]

    analysis_type = analysis_type.upper()

    if analysis_type not in supported_types:
        error_message = f"Unsupported analysis type: {analysis_type}"
        logging.error(error_message)
        raise ValueError(error_message)

    # UMAP
    if analysis_type == "UMAP":
        logging.debug("Using UMAP as dimensionality reduction method")
        reducer = umap.UMAP()

    # TSNE
    elif analysis_type == "TSNE":
        logging.debug("Using tSNE as dimentionality reduction method")
        reducer = TSNE(init = 'pca', random_state = 7)
        # reducer = TSNE(n_components=2, init='pca', random_state=0)
    
    # MDS
    elif analysis_type == "MDS":
        logging.debug("Using MDS as dimentionality reduction method")
        reducer = MDS(n_components = 2)
        
    # PCA
    else: #elif analysis_type = "PCA":
        logging.debug("Using PCA as dimensionality reduction method")
        reducer = PCA(n_components = 2)
    
    X = reducer.fit_transform(data)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})
    return df

# img_to_matrix and flatten_image functions 
# from https://gist.github.com/glamp/5756612
def img_to_matrix(filename):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)

    try:
        img.load()
    except IOError: # this handling is far from ideal
        err_msg = f'{filename} cannot be loaded!'
        logging.error(err_msg) 
        raise IOError(err_msg)	
	    
    logging.debug('Opening %s', filename)
    
    if img.size != STANDARD_SIZE:
        logging.debug('Changing size from %s to %s.', 
                      img.size, STANDARD_SIZE)
        # use PIL to resize to above   
        img = img.resize(STANDARD_SIZE)
    
    logging.debug(str(img.getdata()))

    # Returns the contents of this image as a sequence object 
    # containing pixel values.
    img = list(img.getdata())
    img = np.array(img)

    return img 

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    logging.debug('Flattening the image.')
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

if __name__ == '__main__':
    main()

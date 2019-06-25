DESC="zegamiML - dimensionality reduction applied on the set of images to be
used in Zegami processing"

import os
import argparse
import rasterfairy

def main:
    parser = argparse.ArgumentParser(
            prog = "zegamiML",
            description = DESC
    )
    parser.add_argument(
            '-i', '--input', 
            help = 'Input file name (should be a TSV)',
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
            '-p', '--plot', 
            help = 'Do a plot based on resulting X and Y coords (requires PIL).',
            action = 'store_true', 
            required = False
    )
    args = parser.parse_args()


def test:
    output_file = "zegami_ML.tab"
    images = "./out"
    input_file = "zegami.tab"
    image_column = "image"



from pprint import pprint
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import pandas as pd
#import pylab as pl
import matplotlib.pyplot as plt
import sys

from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import TSNE
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
files = os.listdir(args.image_dir)

# these functions from https://gist.github.com/glamp/5756612
#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (100, 100)
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    
    
    try:
        img.load()
    except IOError:
    	pass
	    
    print "Opening "+filename
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    # use PIL to resize to above   
    img = img.resize(STANDARD_SIZE)
    print str(img.getdata())
    # Returns the contents of this image as a sequence object containing pixel values.
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img 

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)

    #print np.matrix(img_wide[0])
    return img_wide[0]

# read in the zegami table
try:
    zt = pd.read_csv(args.input,sep="\t", header= 0)
except:
    zt = pd.read_csv(args.input,sep=',', header= 0)

# get a list of images via path
images = zt[args.image_column]

data = []
labels = []

#print "Processing a "+str(images)
for image in images:
	image = args.image_dir + "/" + image
	if (os.path.isfile(image)):
	    print "processing %s ",image
	    img = img_to_matrix(image)
	    img = flatten_image(img)
	    data.append(img)

print "Creating PCA or TSNE..."
# unhash the one you want TSNE is more sensitive 
# but plots will be different each time

if (args.analysis_type == "TSNE"): 
	pca = TSNE()
	#pca = TSNE(n_components=2, init='pca', random_state=0)
else:
#or can do PCA
	pca = RandomizedPCA(n_components=2)

X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})
x_label =  args.analysis_type + "_x"
y_label =  args.analysis_type + "_y"
zt[x_label]=df['x']
zt[y_label]=df['y']

print "=========================="
#print str(df)
print "Appended x and y coordinates to "+args.input+" and created "+args.output+"."
zt.to_csv(args.output,sep="\t",index=False)
print "=========================="


# Do a scatter plot, you can leave this out
# especially if you only a terminal with no XWindow
if args.plot:
	pl.scatter(df['x'], df['y'])
	pl.show()



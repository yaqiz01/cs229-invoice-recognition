#!/usr/bin/python2.7
from os import listdir
from os.path import isfile, join, splitext
import os
import argparse
import csv
import subprocess
import commands
import features as ftr 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from util import *
from scipy.stats import entropy
import pickle

def filterFeatureSelection():
    folder = 'experiments/graph/plots/'
    imgs = []
    fig = plt.figure(figsize=(12,4.5), dpi=300)
    # gs1 = gridspec.GridSpec(4, 4)
    # gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    imgs.append(plt.imread('{0}{1}'.format(folder, 'fetsel_TRAINING_ERR_nb_zp0.3.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'fetsel_TRAINING_ERR_logreg_zp0.3.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'fetsel_TRAINING_ERR_svm_zp0.3.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'fetsel_K-FOLD_nb_zp0.3.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'fetsel_K-FOLD_logreg_zp0.3.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'fetsel_K-FOLD_svm_zp0.3.png')))
    # imgs.append(imread('{0}{1}'.format(folder, 'fetsel_TESTING_ERR_logreg_python.png'))
    # imgs.append(imread('{0}{1}'.format(folder, 'fetsel_TESTING_ERR_nb_python.png'))
    # imgs.append(imread('{0}{1}'.format(folder, 'fetsel_TESTING_ERR_svm_python.png'))
    ncol = 3
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs)/ncol, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    plt.savefig('{0}/filterFS.pdf'.format(folder), dpi=fig.dpi, pad_inches=0, bbox_inches="tight")
# model_logreg.png
# model_nb.png
# model_svm.png

def model_all():
    folder = 'experiments/graph/plots/'
    imgs = []
    fig = plt.figure(figsize=(12,4.5), dpi=300)
    imgs.append(plt.imread('{0}{1}'.format(folder, 'model_all_zp0.3_inc434_TRAINING_ERR.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'model_all_zp0.3_inc434_K-FOLD.png')))
    imgs.append(plt.imread('{0}{1}'.format(folder, 'model_all_zp0.0_inc197_K-FOLD.png')))
    ncol = 3
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs)/ncol, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    plt.savefig('{0}/model_all.pdf'.format(folder), dpi=fig.dpi, pad_inches=0, bbox_inches="tight")

def main():
    usage = "Assemble figures for report"
    parser = argparse.ArgumentParser(description='Run feature experiments')
    # parser.add_argument('--matlab', dest='matlab', action='store_true', default=False)
    # parser.add_argument('--mode', dest='mode', action='store', default='K-FOLD')
    # parser.add_argument('--model', dest='model', action='store', default='python')
    # parser.add_argument('--algo', dest='algo', action='store', default='nb')
    # parser.add_argument('--zeropct', dest='zeropct', nargs='?', default=0.3, type=float,
            # help='percentage of zero to include')
    # parser.add_argument('--C', dest='C', nargs='?', default=0.14, type=float,
            # help='penality param C of the regularization error term')
    # parser.add_argument('--k', dest='k', nargs='?', default=10, type=int,
            # help='k for K-Fold')
    (opts, args) = parser.parse_known_args()

    # filterFeatureSelection()
    model_all()
    
if __name__ == "__main__":
    main()

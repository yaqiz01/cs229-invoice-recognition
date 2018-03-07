#!/usr/bin/python2.7

from os import listdir
from os.path import isfile, join, splitext
import os
import csv
import re
from shutil import copyfile
import xml.etree.ElementTree as ET
from porter2 import stem
from dateutil.parser import parse
import argparse
from util import *
import random

VERTICAL_ALIGNMENT_TOLERANCE = 0.01
HORIZONTAL_ALIGNMENT_TOLERANCE = 0.01
#NEARBY_SEARCH_RANGE = 0.2

itemMap = { 'Other':0,
            'Invoice Number':1, 
            'Invoice Date':2, 
            'Total Amount':3,
            'PO #':4,
            'Payment Term':5,
            'Due Date':6,
            'Tax':7
            # 'Shipping':8
        }
itemReverseMap = {}
for item in itemMap:
    itemReverseMap[itemMap[item]] = item

data_path = 'data/'
exp_path = 'experiments/'
training_img_xml_path = '{0}Training_Image_xml/'.format(data_path)
training_input_path = '{0}Training_Input/'.format(exp_path)

def filterFeatures(features, apcount):
    originalNumFeatures = len(features)
    toremove = []
    for f in features:
        (ftext, ftpe) = f
        if (ftpe in ['vAlign', 'hAlign', 'nearby']):
            if (apcount[ftext]<=1):
                toremove.append(f)
    for f in toremove:
        features.remove(f)
    # for t in apcount:
        # if apcount[t]>1:
            # print('{0} {1}'.format(t, apcount[t]))
    # print('Original numFeatures={0}, New numFeatures={1}'.format(originalNumFeatures, len(features)))
    return features

def coordOf(ele):
    [llX, llY, trX, trY] = ele.get('bbox').split(",")
    [llX, llY, trX, trY] = [float(llX), float(llY), float(trX), float(trY)]
    mX = (llX + trX)/2
    mY = (llY + trY)/2
    return (llX, llY, trX, trY, mX, mY) 

def getPageSize(page):
    _,_,width,height,_,_=coordOf(page)
    return (float(width), float(height))

def getId(label, title):
    [pageId, textboxId] = label[title].split(',')
    return (int(pageId), int(textboxId))

def distSqr(x1, y1, x2, y2):
    return pow((x1-x2),2)+pow((y1-y2),2) # use dist^2 for distance. Avoid using sqrt might be faster

def findRelated(tree, target, hTolerance, vTolerance, nearbyRange, fToggle, features):
    (llX0, llY0, trX0, trY0, mX0, mY0) = coordOf(target)
    pages = tree.getroot()
    related                 = {}
    related['vLeftAlign']   = [target]
    related['vRightAlign']  = [target]
    related['vCenterAlign'] = [target]
    related['hTopAlign']    = [target]
    related['hBottomAlign'] = [target]
    related['hCenterAlign'] = [target]
    related['nearby']       = [target]
    for page in pages:
        for textbox in page:
            if (textbox.tag!='textbox'):
                continue
            if (textbox.get('text')==None):
                continue
            if (textbox.get('processed_text')==''):
                continue
            (llX, llY, trX, trY, mX, mY) = coordOf(textbox)
            included = False
            if (abs(llX0-llX) < vTolerance):
                related['vLeftAlign'].append(textbox) 
                included = True
            if (abs(trX0-trX) < vTolerance):
                related['vRightAlign'].append(textbox) 
                included = True
            if (abs(mX0-mX) < vTolerance):
                related['vCenterAlign'].append(textbox) 
                included = True
            if (abs(llY0-llY) < hTolerance):
                related['hBottomAlign'].append(textbox) 
                included = True
            if (abs(trY0-trY) < hTolerance):
                related['hTopAlign'].append(textbox) 
                included = True
            if (abs(mY0-mY) < hTolerance):
                related['hCenterAlign'].append(textbox) 
                included = True
            if (distSqr(mX0, mY0, mX, mY) < pow(nearbyRange,2)):
                related['nearby'].append(textbox) 
                included = True
            if (included):
                ptexts = textbox.get('processed_text').split(" ")
                for ptext in ptexts:
                    pv = (ptext, 'vAlign')
                    if pv not in features and fToggle['vAlign']:
                        features.append(pv)
                    ph = (ptext, 'hAlign')
                    if ph not in features and fToggle['hAlign']:
                        features.append(ph)
                    pn = (ptext, 'nearby')
                    if pn not in features and fToggle['nearby']:
                        features.append(pn)
    # print('target:' + target.get('text'))
    # print('vLeftAlign:' + ', '.join(e.get('text') for e in related['vLeftAlign']))
    # print('vRightAlign:' + ', '.join(e.get('text') for e in related['vRightAlign']))
    # print('vCenterAlign:' + ', '.join(e.get('text') for e in related['vCenterAlign']))
    # print('hTopAlign:' + ', '.join(e.get('text') for e in related['hTopAlign']))
    # print('hBottomAlign:' + ', '.join(e.get('text') for e in related['hBottomAlign']))
    # print('hCenterAlign:' + ', '.join(e.get('text') for e in related['hCenterAlign']))
    # print('nearby:' + ', '.join(e.get('text') for e in related['nearby']))
    return related

def emitCSV(outFile, fToggle, xMap, features):
    with open('{0}{1}'.format(training_input_path, outFile), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ['textbox'] + [u'{0}_{1}'.format(f[0], f[1]) for f in features] + ['item']
        numCol = len(features)+2
        assert(len(header)==numCol)
        writer.writerow([s.encode('utf-8') for s in header])
        keys = sorted(xMap.keys())
        stat = {}
        for item in itemMap:
            stat[item] = 0
        for fileName, pId, tId, item in keys:
            (textbox, related, pos) = xMap[(fileName, pId, tId, item)]
            stat[item] += 1
            dataName = u'fileName={0}_pageId={1}_textboxId={2}_text=\'{3}\'_item={4}'.format(
                    fileName, pId, tId, textbox.get('text'), item)
            dataName = dataName.replace(',', '.')
            row = [dataName]
            ptext = textbox.get('processed_text')
            tpes = textbox.get('type')
            for f in features:
                (word, ftpe) = f
                matched = None
                if ftpe=='vAlign':
                    aligns = []
                    aligns += reduce(lambda x,y:x+y, [tb.get('processed_text').split(' ') for tb in related['vLeftAlign']])
                    aligns += reduce(lambda x,y:x+y, [tb.get('processed_text').split(' ') for tb in related['vRightAlign']])
                    aligns += reduce(lambda x,y:x+y, [tb.get('processed_text').split(' ') for tb in related['vCenterAlign']])
                    if (word in aligns and word not in ptext and fToggle['vAlign']):
                        row.append("1")
                    else:
                        row.append("0")
                elif (ftpe=='hAlign'):
                    aligns = []
                    aligns += reduce(lambda x,y:x+y, 
                            [tb.get('processed_text').split(' ') for tb in related['hTopAlign']])
                    aligns += reduce(lambda x,y:x+y, 
                            [tb.get('processed_text').split(' ') for tb in related['hBottomAlign']])
                    aligns += reduce(lambda x,y:x+y, 
                            [tb.get('processed_text').split(' ') for tb in related['hCenterAlign']])
                    if (word in aligns and word not in ptext and fToggle['hAlign']):
                        row.append("1")
                    else:
                        row.append("0")
                elif (ftpe=='nearby'):
                    nearbys = reduce(lambda x,y:x+y, [tb.get('processed_text').split(' ') for tb in related['nearby']])
                    if (word in nearbys and word not in ptext and fToggle['nearby']):
                        row.append("1")
                    else:
                        row.append("0")
                elif (ftpe=='pos' and fToggle['pos']):
                    if (pos==word):
                        row.append("1")
                    else:
                        row.append("0")
                elif (ftpe=='selfType' and fToggle['selfType']):
                    if (word in tpes):
                        row.append("1")
                    else:
                        row.append("0")
                else:
                    print('Unknown feature type {0}'.format(ftpe))
                    exit(-1)
                # print(u'word={0}'.format(word))
                # print('vLeftAlign:' + ', '.join(e.get('text') for e in related['vLeftAlign']))
                # print('vRightAlign:' + ', '.join(e.get('text') for e in related['vRightAlign']))
                # print('vCenterAlign:' + ', '.join(e.get('text') for e in related['vCenterAlign']))
                # print('hTopAlign:' + ', '.join(e.get('text') for e in related['hTopAlign']))
                # print('hBottomAlign:' + ', '.join(e.get('text') for e in related['hBottomAlign']))
                # print('hCenterAlign:' + ', '.join(e.get('text') for e in related['hCenterAlign']))
                # print('nearby:' + ', '.join(e.get('text') for e in related['nearby']))
            row.append(str(itemMap[item])) # y
            if (len(row)!=numCol):
                print('numCol={0} not equals to numFeatures={1}+2!'.format(len(row), numCol))
                exit(-1)
            writer.writerow([s.encode('utf-8') for s in row])

        print(stat)

def gather(nbrange, outFile, fToggle, **options):
    zeropct = 0.3
    if ('zeropct' in options):
        zeropct = options['zeropct']

    # reset
    features = [] # not using a set to make codegen order deterministic
    xMap = {}
    apcount = {} # appearence count. Count number of appearence of a word accross different invoice

    fileNames = []

    metadata = {}
    input_file = csv.DictReader(open('{0}Training_Data.csv'.format(data_path), mode='r'))
    for row in input_file:
        # print('{0} english={1} standard={2}'.format(row['ID'], english, standard))
        english = row['English?'] in ('yes', 'Yes')
        standard = row['Standard?'] in ('yes', 'Yes')
        #maxFile = int(row['ID'].split('_')[0]) < 10
        maxFile = True 
        if (english and standard and maxFile):
            metadata[row['ID']] = row
            fileNames.append(row['ID'])

    with open('{0}label.csv'.format(data_path), mode='r') as csvfile:
        labels = csv.DictReader(csvfile)
        for label in labels:
            fileName = label['ID']
            if (fileName not in fileNames):
                continue
            # print('fileName=' + fileName)
            itemDict = {} # map between (pageId, textboxId) -> [items]
            for item in label:
                if item=='':
                    continue
                if item!='ID' and item in itemMap:
                    if (label[item]=='NA'):
                        continue
                    (pageId, textboxId) = getId(label, item) 
                    if (pageId, textboxId) not in itemDict:
                        itemDict[(pageId, textboxId)] = [item] 
                    else:
                        itemDict[(pageId, textboxId)].append(item)

            # Gather word counts
            fns = fileName.split('_')
            if (len(fns)==1):
                isFirstPage = True
            elif (fns[1]=='1'):
                isFirstPage = True
            else:
                isFirstPage = False 
            countedWords = [] # counted words in current invoice 

            xmlFile = '{0}{1}.xml'.format(training_img_xml_path, fileName)
            tree = ET.parse(xmlFile)
            pages = tree.getroot()
            assert(pages.get('updated')=='True')
            for page in pages:
                pId = int(page.get('id'))
                (pageWidth, pageHeight) = getPageSize(page)
                hTolerance = pageHeight * HORIZONTAL_ALIGNMENT_TOLERANCE
                vTolerance = pageWidth * VERTICAL_ALIGNMENT_TOLERANCE
                nearbyRange = pageWidth * nbrange 
                for textbox in page:
                    if (textbox.tag!='textbox'):
                        continue
                    ptext = textbox.get('processed_text')
                    if (ptext==''):
                        continue
                    for pt in ptext.split(' '):
                        if pt in apcount and isFirstPage and pt not in countedWords: 
                            apcount[pt] += 1
                            countedWords.append(pt)
                        elif pt not in apcount:
                            apcount[pt] = 1
                            countedWords.append(pt)
                    tId = int(textbox.get('id'))
                    if (pId, tId) not in itemDict:
                        include = random.randint(0,9) < 10 * zeropct
                        if not include:
                            continue
                        itemDict[(pId, tId)] = ['Other']
                    (llX0, llY0, trX0, trY0, mX0, mY0) = coordOf(textbox)
                    if (mY0 < pageHeight * 0.25):
                        pos = '0-0.25'
                    elif (mY0 < pageHeight * 0.5):
                        pos = '0.25-0.5'
                    elif (mY0 < pageHeight * 0.75):
                        pos = '0.5-0.75'
                    else:
                        pos = '0.75-1'
                    related = findRelated(tree, textbox, hTolerance, vTolerance, nearbyRange,
                            fToggle, features)
                    for item in itemDict[(pId, tId)]:
                        xMap[(fileName, pId, tId, item)] = (textbox, related, pos)
    if (fToggle['pos']):
        features.append(('0-0.25', 'pos'))
        features.append(('0.25-0.5', 'pos'))
        features.append(('0.5-0.75', 'pos'))
        features.append(('0.75-1', 'pos'))
    if (fToggle['selfType']):
        features.append(('date', 'selfType'))
        features.append(('money', 'selfType'))
        features.append(('number', 'selfType'))
        features.append(('text', 'selfType'))
    features = filterFeatures(features, apcount)
    emitCSV(outFile, fToggle, xMap, features)

def main():
    usage = "Usage: feature [options --output]"
    parser = argparse.ArgumentParser(description='Gathering features for invoice recognition')
    parser.add_argument('--output', dest='output', action='store', default='training_input.csv',
            help='output traning input name')
    parser.add_argument('--nbrange', dest='nbrange', nargs='?', default=0.05, type=float,
            help='nearby searching range')
    parser.add_argument('--zeropct', dest='zeropct', nargs='?', default=0.3, type=float,
            help='percentage of zero to include')

    (opts, args) = parser.parse_known_args()
    nbrange = opts.nbrange
    outFile = opts.output
    fToggle = { 'vAlign':True, 'hAlign':True, 'nearby':True, 'pos':True, 'selfType':True }
    tic()
    gather(nbrange, outFile, fToggle, zeropct=opts.zeropct)
    toc()
    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table line detect
@author: chineseocr
"""

from utils.config import tableModeLinePath
from utils.utils import letterbox_image,get_table_line,adjust_lines,line_to_line
import numpy as np
import cv2
from nets.unet import unet as Unet

model = Unet((None,None,3),2)
model.load_weights(tableModeLinePath)

def table_line(img,size=(512,512),hprob=0.5,vprob=0.5,row=50,col=10,alph=15):
    sizew,sizeh = size
    inputBlob,fx,fy = letterbox_image(img[...,::-1],(sizew,sizeh))
    pred = model.predict(np.array([np.array(inputBlob)/255.0]))
    pred = pred[0]
    vpred = pred[...,1]>vprob##竖线
    hpred = pred[...,0]>hprob##横线
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)
    colboxes = get_table_line(vpred,axis=1,lineW=col)
    rowboxes = get_table_line(hpred,axis=0,lineW=row)
    # ccolbox = []
    # crowlbox= []
    if len(rowboxes)>0:
       rowboxes = np.array(rowboxes)
       rowboxes[:,[0,2]]=rowboxes[:,[0,2]]/fx
       rowboxes[:,[1,3]]=rowboxes[:,[1,3]]/fy
       # xmin = rowboxes[:,[0,2]].min()
       # xmax = rowboxes[:,[0,2]].max()
       # ymin = rowboxes[:,[1,3]].min()
       # ymax = rowboxes[:,[1,3]].max()
       # ccolbox = [[xmin,ymin,xmin,ymax],[xmax,ymin,xmax,ymax]]
       rowboxes = rowboxes.tolist()
       
    if len(colboxes)>0:
      colboxes = np.array(colboxes)
      colboxes[:,[0,2]]=colboxes[:,[0,2]]/fx
      colboxes[:,[1,3]]=colboxes[:,[1,3]]/fy
      
      # xmin = colboxes[:,[0,2]].min()
      # xmax = colboxes[:,[0,2]].max()
      # ymin = colboxes[:,[1,3]].min()
      # ymax = colboxes[:,[1,3]].max()
      # crowlbox = [[xmin,ymin,xmax,ymin],[xmin,ymax,xmax,ymax]]
      colboxes = colboxes.tolist()
       
    # rowboxes+=crowlbox
    # colboxes+=ccolbox
    print("预测出横线：",len(rowboxes),"竖线：",len(colboxes))

    # rboxes_row_ = adjust_lines(rowboxes, alph=alph)
    # rboxes_col_ = adjust_lines(colboxes, alph=alph)
    # rowboxes +=rboxes_row_
    # colboxes +=rboxes_col_
    # print("调整后横线：", len(rowboxes), "竖线：", len(colboxes))

    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
       for j in range(ncol):
          rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], 10)
          colboxes[j] = line_to_line(colboxes[j], rowboxes[i], 10)

    return rowboxes, colboxes


if __name__ == '__main__':
    import time
    import os
    from utils.utils import draw_lines
    dir = "data/test/"
    files = os.listdir(dir)
    for file in files:
        print(file)
        if file != ".DS_Store":
            img_path = os.path.join(dir,file)
            img = cv2.imread(img_path)
            t = time.time()

            rowboxes, colboxes = table_line(img[...,::-1], size=(1024,1024), hprob=0.5,vprob=0.5,row=50,col=15,alph=15)
            img = draw_lines(img, rowboxes+colboxes, color=(255,0,0), lineW=2)

            print(time.time()-t, len(rowboxes), len(colboxes))
            draw_path = os.path.join("data/11/",file)
            cv2.imwrite(draw_path, img)
# Authors: Pranav Kadam, Anirudh Kulkarni
# Last Updated: 2nd June, 2017
# Programming Language: Python 2.7
# Platform: Raspberry Pi 3, Raspian Jessie OS

import cv2
import os
import argparse
import cv2
#import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


os.system("fswebcam -r 320x240 --no-banner G.jpg")
img1 = cv2.imread("G.jpg")
img1 = cv2.rotate(img1,rotateCode=1)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be Preprocessed")
args = vars(ap.parse_args())

value = 100
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
ratio = 3
kernel_size = 3


img = cv2.imread(args["image"])
img = cv2.rotate(img,rotateCode=1)

img = cv2.medianBlur(img,3)
img = cv2.medianBlur(img,3)

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,3)
blur = cv2.medianBlur(blur,3)

ret,thresh = cv2.threshold(blur,value,255,cv2.THRESH_BINARY)

thresh = cv2.medianBlur(thresh,3)

open1 = cv2.dilate(thresh,kernel,iterations=2)
close = cv2.erode(open1,kernel,iterations=3)

detected_edges = cv2.medianBlur(close,3)
edge = cv2.Canny(detected_edges,0,0*ratio,apertureSize = kernel_size)

(r,c) = edge.shape

im2, contours, hierarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
length = len(contours)
len0 = len(contours[0])
x = 0
for l in range (1,length):
	if len0 < len(contours[l]):
		len0 = len(contours[l])
		x = l
con = contours[x]

img3 = np.zeros((r,c))

k = cv2.drawContours(img3, [con], 0 , 255, 1)

for col in range (0,c):
	if k[r-2,col] == 255:
		d = col
		break
		
c = r-1
len1=len(con)
a = con[0][0]
q = len(con)

dst = []
xaxis = []

for index in range (0,((q-1)/5)): 	
	dst.append(distance.euclidean([d,c],con[index][0]))
	xaxis.append(index)
w = max(dst)

dst[:] = [x / w for x in dst]

a1 = dst

#Store all templates here

distance1,path1 = fastdtw(a1,template1,dist=euclidean)
print distance1

distance2,path2 = fastdtw(a1,atemplate2,dist=euclidean)
print distance2 

distance3,path3 = fastdtw(a1,template3,dist=euclidean)
print distance3 

distance4,path4 = fastdtw(a1,template4,dist=euclidean)
print distance4 

distance5,path5 = fastdtw(a1,template5,dist=euclidean)
print distance5

distance6,path6 = fastdtw(a1,template6,dist=euclidean)
print distance6

distance7,path7 = fastdtw(a1,template7,dist=euclidean)
print distance7

distance8,path8 = fastdtw(a1,template8,dist=euclidean)
print distance8

g = [distance1,distance2,distance3,distance4,distance5,distance6,distance7,distance8]

o = 0
for i in range(8):
	if g[i] == min(g) :
		o = i+1

if o == 1:
	print "one Gesture"
	os.system("omxplayer -o local Voice1.wav")
	
if o == 2:
	print "Two Gesture"
	os.system("omxplayer -o local Voice2.wav")

if o == 3:
	print "Three Gesture"
	os.system("omxplayer -o local Voice3.wav")

if o == 4:
	print "Four Gesture"
	os.system("omxplayer -o local Voice4.wav")

if o == 5:
	print "Five Gesture"
	os.system("omxplayer -o local Voice5.wav")

if o == 6:
	print "C Gesture"
	os.system("omxplayer -o local VoiceC.wav")

if o == 7:
	print "Y Gesture"
	os.system("omxplayer -o local VoiceY.wav")

if o == 8:
	print "A Gesture"
	os.system("omxplayer -o local VoiceA.wav")

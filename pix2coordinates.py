
import gdal
import os
import cv2

def ConvertPixel2XY (obj, x1, y1):

    
    cordArr = str(obj).split(",");
    xy_list = [];
    
    try:
        for x in range(len(cordArr)): 
            print(cordArr[x]) 
            x,y = (cordArr[x].split(" "));
            
            xdiff = x1 + (float(x) / 50);
            ydiff = y1 + ((2500 - float(y)) / 50);
            
            xy_list.append( str(xdiff) + ' ' + str(ydiff));
            
    except:
        print(obj);
        conv = False
         
        
    return xy_list, conv;

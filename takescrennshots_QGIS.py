Steps to take the screenshots from the satillite imagery in QGIS

Create the polygon layer on the required area from the entire satillite imagery 
Create the grids inside the polygon
Make the grid visibility transperant 
Run the below script to take the screenshots 

Note:before runnig the below script make sure the grid layer is active layer 


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from qgis.core import *
import qgis.utils
from qgis.utils import iface
from qgis.gui import *
import sys
import os
import ctypes
import processing
import time

layer = iface.activeLayer()
ids = layer.allFeatureIds()
counter = 0
def exportMap():
    global ids
    QgsProject.instance().layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(False)
    time.sleep(1)
    iface.mapCanvas().saveAsImage( "PATH\\XXX{}.jpg".format( ids.pop() ) )
    if ids:
        QgsProject.instance().layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(True)
        setNextFeatureExtent()
    else: 
        iface.mapCanvas().mapCanvasRefreshed.disconnect( exportMap )

def setNextFeatureExtent():
    iface.mapCanvas().zoomToFeatureIds( layer, [ids[-1]] )
    #it = layer.getFeature( ids[-1] )
    #newRect = QgsRectangle (float(it[0]), float(it[3]), float(it[2]), float(it[3]))
    #iface.mapCanvas ().setExtent (newRect)
    #iface.mapCanvas ().refresh()
        
iface.mapCanvas().mapCanvasRefreshed.connect( exportMap )
setNextFeatureExtent() # Let's start
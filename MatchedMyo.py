'''
First pass at making a GUI for the matched filtering algorithm.
  This is to simplify the input and make the algorithm more approachable to
  everyday users.
'''

from Tkinter import *
import tkFileDialog
import util

### Making inputs class
class empty: pass
inputs = empty()

### defining main window
root = Tk()
root.title("MatchedMyo")
root.minsize(1000,500)
root.geometry("1000x500")

### Make Function to Select File Name
def selectFileName():
  inputs.fileName = tkFileDialog.askopenfilename(initialdir="./",
                                                 title="Select Myocyte Image File",
                                                 filetypes=(("PNG","*.png"), # add other accepted file types
                                                            ("TIFF","*.tif")))
  try:
    _ = util.ReadImg(inputs.fileName)
    print "Image was successfully read in"
  except:
    raise RuntimeError("Image was not read in correctly")

### Make File Dialogue Button for Image Name
fileButton = Button(root, text="Image Selection", command=selectFileName,height=50,width=100,compound=LEFT)
fileButton.place(x=50,y=50)

### Call myocyteFigs.giveMarkedMyocyte with inputs

mainloop()

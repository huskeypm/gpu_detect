'''
First pass at making a GUI for the matched filtering algorithm.
  This is to simplify the input and make the algorithm more approachable to
  everyday users.

AUTHOR: Dylan F. Colli
INCEPTION: July 21, 2018
'''

from appJar import gui
import myocyteFigs as mF
import preprocessing as pp

### create GUI variable with specified name
app = gui("MatchedMyo")

### specify background color
app.setBg("white")

### specify font size
app.setFont(12)

### specify the window can't be resized
app.setResizable(False)

### specify image path or directory
rowNumber=app.getRow()
app.addLabel("1.","1. Image Path",rowNumber,0)
rowNumber = app.getRow()
app.addHorizontalSeparator(rowNumber,0,4,colour="black")
app.addFileEntry("fileName")
rowNumber = app.getRow()
app.addHorizontalSeparator(rowNumber,0,4,colour="black")

### create space
app.addLabel("space1","")
app.addLabel("space2","")


### check which preprocessing routines that are desired
app.addLabel("2.","2. Preprocessing Routines")
rowNumber = app.getRow()
app.addHorizontalSeparator(rowNumber,0,4,colour="black")
rowNumber = app.getRow()
app.addCheckBox("Reorientation",rowNumber,0)
app.setCheckBox("Reorientation",ticked=True)
app.addCheckBox("Resizing",rowNumber,1)
app.setCheckBox("Resizing", ticked=True)
app.addCheckBox("CLAHE",rowNumber,2)
app.setCheckBox("CLAHE", ticked=True)
app.addCheckBox("Striation Normalization",rowNumber,3)
app.setCheckBox("Striation Normalization", ticked=True)
#app.addCheckBox("Mask Construction",rowNumber,4) # not implemented yet
rowNumber = app.getRow()
app.addHorizontalSeparator(rowNumber,0,4,colour="black")

### create space
app.addLabel("space3","")
app.addLabel("space4","")


### Take in Algorithm Inputs
app.addLabel("3.","3. Algorithm Inputs")
rowNumber = app.getRow()
app.addHorizontalSeparator(rowNumber,0,4,colour="black")
## get filter two sarcomere size (used to resize the image)
rowNumber = app.getRow()
app.addLabel("FTSS","Filter Two Sarcomere Size (pixels)",rowNumber,0)
app.addNumericEntry("FiltSarcSize",rowNumber,1)
app.setEntry("FiltSarcSize",25,callFunction=False)
## get filtering modes that you want marked
rowNumber = app.getRow()
app.addCheckBox("WT Filtering",rowNumber,0)
app.setCheckBox("WT Filtering",ticked=True)
app.addCheckBox("LT Filtering",rowNumber,1)
app.setCheckBox("LT Filtering",ticked=True)
app.addCheckBox("TA Filtering",rowNumber,2)
app.setCheckBox("TA Filtering",ticked=True)

### Set filtering modes for each filter
#rowNumber = app.getRow()
#app.addLabel("WTfiltmode","WT Filtering Mode",rowNumber,0)
#app.addLabel("LTfiltmode","LT Filtering Mode",rowNumber,1)
#app.addLabel("TAfiltmode","TA Filtering Mode",rowNumber,2)

#rowNumber=app.getRow()
#options = ["Simple Detction", "Punishment Filter", "Regional Deviation"
#app.addOptionBox("WTfilt",

### select paths to the filters needed
rowNumber = app.getRow()
app.addLabel("WT Filter Path","WT Filter Path",rowNumber,0)
app.addLabel("LT Filter Path","LT Filter Path",rowNumber,1)
app.addLabel("TA Filter Path","TA Filter Path",rowNumber,2)
rowNumber = app.getRow()
app.addFileEntry("WTfilter",rowNumber,0)
app.setEntry("WTfilter","./myoimages/newSimpleWTFilter.png",callFunction=False)
app.addFileEntry("LTfilter",rowNumber,1)
app.setEntry("LTfilter","./myoimages/LongitudinalFilter.png",callFunction=False)
app.addFileEntry("TAfilter",rowNumber,2)
app.setEntry("TAfilter","./myoimages/LossFilter.png",callFunction=False)
rowNumber = app.getRow()
app.addFileEntry("WTpunishfilter",rowNumber,0)
app.setEntry("WTpunishfilter","./myoimages/newSimpleWTPunishmentFilter.png",callFunction=False)

### add in boxes for thresholds and filtering parameters
rowNumber = app.getRow()
app.addNumericLabelEntry("WT SNR Threshold",rowNumber,0)
app.setEntry("WT SNR Threshold",0.35,callFunction=False)
app.addNumericLabelEntry("LT SNR Threshold",rowNumber,1)
app.setEntry("LT SNR Threshold",0.6,callFunction=False)
app.addNumericLabelEntry("TA SNR Threshold",rowNumber,2)
app.setEntry("TA SNR Threshold",0.04,callFunction=False)

rowNumber = app.getRow()
app.addNumericLabelEntry("WT Punishment Parameter",rowNumber,0)
app.setEntry("WT Punishment Parameter",3.0,callFunction=False)
app.addNumericLabelEntry("LT Standard Deviation Threshold",rowNumber,1)
app.setEntry("LT Standard Deviation Threshold",0.2,callFunction=False)
app.addNumericLabelEntry("TA Standard Deviation Threshold",rowNumber,2)
app.setEntry("TA Standard Deviation Threshold",0.1,callFunction=False)
rowNumber = app.getRow()
app.addHorizontalSeparator(rowNumber,0,4,colour="black")

### create space
app.addLabel("space5","")
app.addLabel("space6","")

### Set whether to use GPU for acceleration
app.addCheckBox("Use GPU for Acceleration")

### create space
app.addLabel("space7","")
app.addLabel("space8","")

### add button to run the matchedmyo program
def runProgram(runnerButton):
  if runnerButton == "Run Program":
    runAnalysis()
  elif runnerButton == "Quit":
    app.stop()
  elif runnerButton == "Restore Defaults":
    print "Function not implemented yet"

rowNumber = app.getRow()
colSpan = 4
app.addButtons(["Restore Defaults","Run Program","Quit"],runProgram,rowNumber,0,colSpan)

### add function to analyze myocyte
def runAnalysis():
  ## get all information from inputs to use in the analysis
  allEntries = app.getAllEntries()

  ## necessary to convert filter two sarcomere size argument to an integer
  allEntries['FiltSarcSize'] = int(allEntries['FiltSarcSize'])

  #for key,value in allEntries.iteritems():
  #  print key,value

  ## preprocess image for use in the algorithm
  preprocessedImage = pp.preprocess(allEntries['fileName'],allEntries['FiltSarcSize'])

  processedFileName = allEntries['fileName'][:-4] + "_processed" + allEntries['fileName'][-4:]

  ## TODO: Need to add in option to return the angle analysis of this too
  ## TODO: Refactor code s.t. we can use the threshes that have been indicated herein
  coloredImage = mF.giveMarkedMyocyte(ttFilterName = allEntries['WTfilter'],
                                      ltFilterName = allEntries['LTfilter'],
                                      lossFilterName = allEntries['TAfilter'],
                                      wtPunishFilterName = allEntries['WTpunishfilter'],
                                      testImage = processedFileName,
                                      tag = allEntries['fileName'][:-4],
                                      writeImage = True
                                      )
                                      


### run the GUI
app.go()

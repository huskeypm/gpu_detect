'''
First pass at making a GUI for the matched filtering algorithm.
  This is to simplify the input and make the algorithm more approachable to
  everyday users.

AUTHOR: Dylan F. Colli
INCEPTION: July 21, 2018
'''

from appJar import gui

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
app.setEntryDefault("FiltSarcSize",25)
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
app.setEntryDefault("WTfilter","./myoimages/newSimpleWTFilter.png")
app.addFileEntry("LTfilter",rowNumber,1)
app.setEntryDefault("LTfilter","./myoimages/LongitudinalFilter.png")
app.addFileEntry("TAfilter",rowNumber,2)
app.setEntryDefault("TAfilter","./myoimages/LossFilter.png")
rowNumber = app.getRow()
app.addFileEntry("WTpunishfilter",rowNumber,0)
app.setEntryDefault("WTpunishfilter","./myoimages/newSimpleWTPunishmentFilter.png")

### add in boxes for thresholds and filtering parameters
rowNumber = app.getRow()
app.addNumericLabelEntry("WT SNR Threshold",rowNumber,0)
app.setEntryDefault("WT SNR Threshold",0.35)
app.addNumericLabelEntry("LT SNR Threshold",rowNumber,1)
app.setEntryDefault("LT SNR Threshold",0.6)
app.addNumericLabelEntry("TA SNR Threshold",rowNumber,2)
app.setEntryDefault("TA SNR Threshold",0.04)

rowNumber = app.getRow()
app.addNumericLabelEntry("WT Punishment Parameter",rowNumber,0)
app.setEntryDefault("WT Punishment Parameter",3.0)
app.addNumericLabelEntry("LT Standard Deviation Threshold",rowNumber,1)
app.setEntryDefault("LT Standard Deviation Threshold",0.2)
app.addNumericLabelEntry("TA Standard Deviation Threshold",rowNumber,2)
app.setEntryDefault("TA Standard Deviation Threshold",0.1)
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
    print "insert runner function here"
  elif runnerButton == "Quit":
    app.stop()
  elif runnerButton == "Restore Defaults":
    print "Function not implemented yet"

rowNumber = app.getRow()
colSpan = 4
app.addButtons(["Restore Defaults","Run Program","Quit"],runProgram,rowNumber,0,colSpan)

### add buttons to restore defaults or quit the GUI
#def quitApp(button):
#  if button == "Quit":
#    app.stop()
#  elif button == "Restore Defaults":
#    print "Function not implemented yet"
#    # use the app.setEntryDefault(title,text) for all text entries created

#app.addButtons(["Restore Defaults","Quit"],quitApp)

### run the GUI
app.go()

### crop images for figure generation
echo 'uncomment for submission'
python myocyteFigs.py -shaveFig fig3_Raw.png
python myocyteFigs.py -shaveFig fig3_ColoredImage.png
python myocyteFigs.py -shaveFig fig3_ColoredAngles.png
python myocyteFigs.py -shaveFig fig3_BarChart.png
python myocyteFigs.py -shaveFig fig3_angle_histogram.png

# for some INCREDIBLY annoying reason the geometry argument squares the resizing factor
# so if you want to scale by 50%, you have to scale by 50%^2 = 70.7%

### draw labels on images
montage -pointsize 100 \
        -gravity NorthWest \
        -draw 'text 25,0 A' fig3_Raw.png \
        -geometry 100% \
        fig3_Raw_marked.png

montage -pointsize 100 \
        -gravity NorthWest \
        -draw 'text 25,0 B' fig3_ColoredImage.png \
        -geometry 100% \
        fig3_ColoredImage_marked.png

montage -pointsize 100 \
        -gravity NorthWest \
        -draw 'text 25,0 C' fig3_ColoredAngles.png \
        -geometry 100% \
        fig3_ColoredAngles_marked.png

montage -pointsize 200 \
        -gravity NorthWest \
        -draw 'text 25,0 D' fig3_BarChart.png \
        -geometry 70.7% \
        fig3_BarChart_marked.png

montage -pointsize 200 \
        -gravity NorthWest \
        -draw 'text 25,0 E' fig3_angle_histogram.png \
        -geometry 70.7% \
        fig3_angle_histogram_marked.png

montage fig3_Raw_marked.png \
        fig3_ColoredImage_marked.png \
        fig3_ColoredAngles_marked.png \
        -tile 1x3 \
        -geometry 100% \
         merged1.png

#        -pointsize 100 \
#        -draw 'text 25,0 A' \
#        -draw 'text 25,935 B' \
#        -draw 'text 25,1870 C' \
#        -tile 1x3 \
#        -geometry 100% \
#         merged1.png


### montage first picture for figure 3
#montage -label 'A' fig3_Raw.png \
#        -label 'B' fig3_ColoredImage.png \
#        -label 'C' fig3_ColoredAngles.png \
#        -tile 1x3 \
#        -geometry 1000 \
#        -pointsize 75 \
#         merged1.png

montage fig3_BarChart_marked.png \
        fig3_angle_histogram_marked.png \
        -tile 2x1 \
        -geometry 100% \
        merged2.png

### montage second picture for figure 3
#montage -label 'D' fig3_BarChart.png \
#        -label 'E' fig3_angle_histogram.png \
#        -tile 2x1 \
#        -geometry 500 \
#        -pointsize 75 \
#        -gravity NorthWest \
#        merged2.png

### montage together for figure 3
montage -mode Concatenate \
        -gravity NorthWest \
        merged1.png \
        merged2.png \
        -tile 1x2 \
        -geometry 100% \
        fig3.png

### I'm ashamed of having to do this
python myocyteFigs.py -shaveFig fig3.png 20 20 0

echo 'Wrote figure 3 as: fig3.png'

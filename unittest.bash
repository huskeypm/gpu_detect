# if running for this first time
# python util.py -genAllMyo
# Run these before committing 

## figures from silica paper 
python silicaFigs.py -validate
python optimizer.py -optimizeLight  # move these to silica figs

## myocyte figures 
python myocyteFigs.py -roc                   
python myocyteFigs.py -minorValidate
python myocyteFigs.py -validate
echo "Broken until longitudinal filtering sorted out";python myocyteFigs.py -scoretest
python tissue.py -validation

## test of comamnd line operation 
python detect.py -validation

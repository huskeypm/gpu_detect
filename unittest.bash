# Run these before committing 
python silicaFigs.py -validate
python myocyteFigs.py -roc                   
python myocyteFigs.py -minorValidate
python myocyteFigs.py -validate
python myocyteFigs.py -scoretest
python tissue.py -validate
python optimizer.py -optimizeLight  # move these to silica figs


python detect.py -validation

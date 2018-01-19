# Run these before committing 
python silicaFigs.py -validate
python myocyteFigs.py -roc                   
echo "broke"; python myocyteFigs.py -minorValidate
echo "broke"; python myocyteFigs.py -validate
python tissue.py -validate
python optimizer.py -optimizeLight  # move these to silica figs



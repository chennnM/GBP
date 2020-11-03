python -u multilabel.py --data ppi --layer 4 --alpha 0.3 --hidden 2048 --batch 2048 --rmax 5e-7 --rrz 0.0 --patience 400
python -u multilabel.py --data yelp --layer 4 --alpha 0.9 --hidden 2048 --batch 30000 --rmax 5e-7 --rrz 0.3 --patience 50
python -u multiclass.py --data Amazon2M --layer 4 --alpha 0.2 --rmax 1e-7 --rrz 0.2 --hidden 1024 --batch 100000 --patience 100
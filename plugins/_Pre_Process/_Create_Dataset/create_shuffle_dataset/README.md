# make virtual environment
```
python -m venv path_to_env
source path_to_env/bin/activate
```

# create shuffle label dataset
```
python create_{dataset_name}_csv.py --label_shuffle
```

generated `{dataset_name}_training_shuffle.csv` in current directory. 

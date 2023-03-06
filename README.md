```
python source/extract_features.py dataset/ local/features/
```

```
python source/create_annotations.py local/features/ local/annotations.csv
```

```
python test.py local/annotations.csv local/features/ config.yml
```

# Reproducibility

Run a syntax smoke check with:
```bash
python -m compileall -q src scripts
```

Full training requires the external dataset and substantial compute. Match the paper dataset split, preprocessing, horizon, seed, and hardware assumptions.

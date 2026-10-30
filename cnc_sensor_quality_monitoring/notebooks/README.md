# Notebooks

Exploratory notebooks mirror pipeline stages. For reproducible EngD results, prefer:

```bash
python run_pipeline.py --stage all
python scripts/generate_reports.py
```

Each `0x_*.ipynb` notebook loads `configs/config.yaml` and calls the corresponding `src/` module so interactive analysis stays aligned with the automated pipeline.

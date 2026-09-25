# Per-file SCAPS results

Every root-level `.iv` file is exported as a clean CSV with one row per I-V scan point, then evaluated independently. Open the matching source-name folder to see `<source>_cleaned.csv`, the per-simulation `dataset.csv`, per-model metrics and predictions, model error graphs, and comparison charts.

`model_metrics_by_file.csv` lists each file/model combination, while `source_file_summary.csv` lists the best model by MAE for each file.

The target is per-run SCAPS solar-cell efficiency (`eta (%)`), not QE versus wavelength. Each file has its own grouped 80/20 split. Small sweeps can produce unstable metrics; the estimates do not establish performance on new materials or parameter regimes.

Original `.iv` files remain in the project as source inputs; generated data and model outputs here are CSV/PNG/TXT.

## Collected charts

All charts are copied into `results/images/`, organized by source file, model,
and chart type. The combined-data charts and earlier QE workflow charts have
separate folders. `image_index.csv` provides a searchable source/model/chart
mapping. Regenerate the collection after updating results with:

```bash
./.venv/bin/python ml_models/collect_images.py
```

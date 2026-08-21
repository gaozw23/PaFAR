# Clinical data

The retrospective analysis uses the public training data from [Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/), version 1.0.0 (DOI: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)). Source clinical records are not redistributed in this repository.

Obtain the data from PhysioNet and place the two training sets at:

```text
data/
├── README.md
└── physionet2019/
    └── raw/
        ├── training_setA/
        └── training_setB/
```

All content under `data/` other than this README remains local and ignored by Git, including raw patient records, processed patient-level data, manifests, feature caches, NumPy arrays, and fitted objects.

From the repository root, validate the local layout without running the analysis:

```bash
python paper_repro/run_realdata.py --check-only
```

Run the maintained real-data pipeline only after the required local files are present:

```bash
python paper_repro/run_realdata.py --stage all --resume --confirm RUN_PAFAR_REALDATA_PRIMARY
```

The wrapper uses `configs/realdata_primary.yaml` and delegates to `scripts/run_realdata_pipeline.py`. It does not download or modify the source PSV files.

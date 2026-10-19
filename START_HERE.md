# Start here: assemble the complete project

The assembled folder should contain `README.md`, `app.py`, `movie_roi/`, `artifacts/`, `data/`, and `tests/`. `data/posters/` must contain 2,601 JPEGs. `data/raw/movies_complete.csv` is the complete original snapshot used by the source audit test. Core contains normalized metadata, current and initial trained artifacts, documentation, experiment records, tests, and the release hash manifest.

Open `README.md` for Python 3.11/3.12 environment setup and launch instructions. Quick start from inside the assembled project folder:

```sh
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

The cached data and models run locally without dataset downloads. Read `QA_REPORT.md`, `EVALUATION.md`, and `DATA_SOURCES.md` for tested behavior and limitations. Posters are third-party copyrighted artwork supplied for this attributed educational prototype; do not publish or commercially redistribute the package without checking applicable rights. This is a box-office-to-budget research proxy, not investment advice.

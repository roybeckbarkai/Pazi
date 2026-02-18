# Pazi

## Streamlit SAXS WLC Explorer

Run the app locally with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-streamlit-saxs.txt
streamlit run streamlit_saxs_wlc_explorer.py
```

### Common Streamlit errors and fixes

- **`streamlit: command not found`**
  - Install dependencies in an activated virtual environment:
    `pip install -r requirements-streamlit-saxs.txt`
  - Launch via module form if PATH is not updated:
    `python -m streamlit run streamlit_saxs_wlc_explorer.py`

- **`ModuleNotFoundError: No module named 'streamlit'`**
  - Ensure the interpreter running the command is the same one where you installed packages:
    `which python && python -m pip show streamlit`

- **`sasmodels` missing at runtime**
  - Install with:
    `pip install sasmodels`
  - The app will still load, but intensity curves require `sasmodels`.

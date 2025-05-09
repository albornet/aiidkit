# AIIDKIT project

AIIDKIT uses machine learning / deep learning to predict post-transplant infection from kidney transplant recipient data. It aims to provide personalized and interpretable infection risk assessments.

## Dependencies

To install dependencies, run the following:
```
wget -qO- https://astral.sh/uv/install.sh | sh  # download uv, a package manager
uv venv && source .venv/bin/activate  # create a local virtual environment
uv pip install numpy pandas tqdm ipdb  # install dependencies
```

## Usage
To pre-process patient data and create patient data records in data/results, run the following:
```
python main.py
```

You can use debug mode to pre-process the dataset for a few patients, using a single core:
```
python main.py -d
```

You use exploration mode to check the different data sheets in the raw data file:
```
python main.py -e
```
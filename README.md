# Take home test for the decision scientist position at Monzo

## Running the code

### Install the environment
This code was created using the [Poetry](https://python-poetry.org/) package management.
To install Poetry follow the description at [poetry installation](https://python-poetry.org/docs/#installation).
Once you have installed Poetry you need to initiate the project:

```bash
poetry init
```

```bash
poetry update
```

Open a shell with the virtual environment loaded:

```bash
poetry shell
```
### Create kernel to use on Jupyter

```bash
python -m ipykernel install --user --name monzo --display-name "monzo"
```

### Open Jupyter Notebook

```bash
jupyter-notebook Monzo_Take_Home_Test_Decision_Scientist_Rafael_Garcia_Dias.ipynb &
```

## Running scripts directly

Everything on the Jupyter Notebook is based on the code development at the monzo_decision_scientist.
The scripts can be run directly by

```bash
python -m <model_path>
```

For example, to clean and explore the data one can run:

```bash
python -m monzo_decision_scientist.data.exploratory_analysis
```
### run all
I have also prepared a bash script to run all the code in sequence.

```bash
bash run_all.sh
```

# Tests
The tests were developed with pytest.
Run all the tests with:

```bash
pytest -svv
```

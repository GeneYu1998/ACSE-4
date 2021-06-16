# ACSE-4-armageddon

This is a very brief Readme to familiarise you with how to install and run the tool you will develop. You should expand and elaborate on this in your final version.

## Installation

To install the module and any pre-requisites
```
pip install -r requirements.txt
```  

## Downloading postcode data

To download the postcode data
```
python download_data.py
```

## Automated testing

To run the pytest test suite
```
python -m pytest armageddon
```

Note that you should keep the tests provided, adding new ones as you develop your code. If any of these tests fail it is likely that the scoring algorithm will not work.

## Documentation

To generate the documentation (in html format)
```
python -m sphinx docs html
```

See the `docs` directory for the preliminary documentation provided that you should add to.

## Example usage

For example usage see `example.py`
```
python example.py
```

## More information

For more information on the project specfication, see the python notebooks: `ProjectDescription.ipynb`, `AirburstSolver.ipynb` and `DamageMapper.ipynb`.

## How to use the package to reproduce the model and get predictions

1. Create a conda environment, with rdkit, and install the package.

```bash
conda create -n qctime -c rdkit rdkit
conda activate qctime
cd models/qc_time_estimator
python setup.py install
# or for development
pip install -e .
```

2. Train the model to generate and save the model's pkl file (used for prediction and tests)

```bash
python models/qc_time_estimator/qc_time_estimator/train_pipeline.py 
```

3. Next, after generating the model, you can run tests (optional)

```bash
cd models/qc_time_estimator
pytest
```

4. Finally, you can run predictions in Python:

Input file example:
```csv
nthreads,driver,method,restricted,cpu_clock_speed,cpu_launch_year
12,gradient,b3lyp,True,2500.0,2014
4,energy,wb97x-d,True,2100.0,2016
8,gradient,b3lyp,True,2200.0,2016
4,energy,pbe,True,2500.0,2014
2,energy,hf,True,2600.0,2013
4,energy,wb97x-d,True,2100.0,2016
16,energy,pbe,True,2100.0,2016
4,gradient,b3lyp,True,2500.0,2014
```

```python
from qc_time_estimator.predict import make_prediction
import pandas as pd

test_input = pd.read_csv('path/to/file.csv')
predictions = make_prediction(input_data=test_input)
```
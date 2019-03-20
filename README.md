

Training one model per TF family:

```
python cross_validate.py
```

Plots will be saved to `report/`.




Training one model for all TF families:

```
python cross_validate_one_model.py
```

Plots will be saved to `report/one_model/`.


TODOs:

- other low throughput experimental data from the paper as test data?

- Nested CV, actual test data

- TF DNA-binding site sequence as input, instead of multi-head output

- filter visualization

- concentration as input to the model, or train as different output?

- make sure there is no duplicated sequence across TF families, if there are, merge the data instead of concatenating (duplicating training example)


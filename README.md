# Cell Models Package

This package is meant to contain all Christini Lab models that are implemented in Python. As of writing this, there are 3 models: O'Hara Rudy, Paci, and Kernik.

### Using and Installing this Project

This Python project is written in [Python Packaging format](https://packaging.python.org/tutorials/packaging-projects/). If you want to play around with the models quickly, you can `cd` into `cell-models/` and edit the `main.py` file. 

If you want to install these models locally, you can run `python3 setup.py` from the project root. This will make the files in `cell-models/` globally available. 

### Tutorial

Much of the code below can be found in `cell-models/main.py`. 

#### Instantiate and generate a response from models

The following will plot spontaneous Kernik and Paci models and a paced O'Hara Rudy model.

```py
# Spontaneous / Stimulated
KERNIK_PROTOCOL = protocols.SpontaneousProtocol(2000)
kernik_baseline = KernikModel()
tr_b = kernik_baseline.generate_response(KERNIK_PROTOCOL)
plt.plot(tr_b.t, tr_b.y)
plt.show()

PACI_PROTOCOL = protocols.SpontaneousProtocol(2)
paci_baseline = PaciModel()
tr_bp = paci_baseline.generate_response(PACI_PROTOCOL)
plt.plot(tr_bp.t, tr_bp.y)
plt.show()

OHARA_RUDY = protocols.PacedProtocol(model_name="OR")
or_baseline = OharaRudyModel()
tr = or_baseline.generate_response(OHARA_RUDY)
plt.plot(tr.t, tr.y)
plt.show()
```

#### Update parameters for a model
The code below will plot the baseline Kernik model and an example model with new parameter values.

```py
KERNIK_PROTOCOL = protocols.SpontaneousProtocol(2000)
kernik_baseline = KernikModel()
kernik_updated = KernikModel(
        updated_parameters={'G_K1': 1.2, 'G_Kr': 0.8, 'G_Na':2.2})
tr_baseline = kernik_baseline.generate_response(KERNIK_PROTOCOL)
tr_updated =  kernik_updated.generate_response(KERNIK_PROTOCOL)
plt.plot(tr_baseline.t, tr_baseline.y, label="Baseline")
plt.plot(tr_updated.t, tr_updated.y, label="Updated")
plt.legend()
plt.show()
```

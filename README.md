# classification-models
BITS assignment of semester 1 assignemnt 2 (implementation of different classification models)

# Dataset
https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits/data

# Streamlit evaluation app

Run the app locally with:

```bash
streamlit run app.py
```

The app evaluates uploaded, labeled test CSV files against every `.pkl` or
`.joblib` artifact in `model/`. It does not train models. The CSV must contain
the target column `sleep_debt_category` and every input feature expected by
each model. A model artifact can override the target column by providing
metadata.

Train all supported models offline with:

```bash
python train_model.py
```

Train only one model by passing its name to `--model`:

```bash
python train_model.py --model decision_tree
```

Supported names are `decision_tree`, `gaussian_naive_bayes`,
`k_nearest_neighbors`, `logistic_regression`, and `support_vector_machine`.
Use `--model-dir` to choose where artifacts are saved; it does not select
which model is trained. Training dispatches to the selected class in
`model_implementation`, calls its `fit()` and `evaluate()` methods, and saves
the fitted estimator together with the preprocessing state needed for inference.
Categorical inputs are one-hot encoded using categories learned from training
rows only; continuous numeric features are scaled using training rows only.

For reliable inference, save a fitted scikit-learn estimator or `Pipeline`
that includes all preprocessing learned from the training data. A Pipeline
fitted with a pandas DataFrame exposes its expected feature names automatically.
If the artifact does not expose feature names, save it as a dictionary with
`model` and `feature_names` entries; it may also include `target_column` and a
fitted `label_encoder` when the model was trained on encoded target labels:

```python
import joblib

joblib.dump(
    {
        "model": fitted_pipeline,
        "feature_names": feature_names,
        "target_column": "sleep_debt_category",
        # "label_encoder": fitted_label_encoder,  # when applicable
    },
    "model/my_classifier.pkl",
)
```

Only deploy model artifacts from trusted sources because loading pickle-based
files can execute code.

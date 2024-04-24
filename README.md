![](https://img.shields.io/github/actions/workflow/status/sumanthprabhu/DQC-Toolkit/test.yml) ![](https://img.shields.io/website?url=https%3A%2F%2Fsumanthprabhu.github.io%2FDQC-Toolkit%2F&label=docs
) ![](https://img.shields.io/pypi/pyversions/DQC-Toolkit) ![](https://img.shields.io/pypi/v/DQC-Toolkit) ![](https://img.shields.io/pypi/l/DQC-toolkit)

![](/docs/images/dqc-toolkit.svg)


DQC Toolkit is a Python library and framework designed with the goal to facilitate improvement of Machine Learning models by identifying and mitigating label errors in training dataset. Currently, DQC toolkit offers `CrossValCurate` for curation of text classification datasets (binary / multi-class) using cross validation based selection.

## Installation

Installation of DQC-toolkit can be done as shown below
```python
pip install dqc-toolkit
```

## Quick Start

 Assuming your text classification data is stored as a pandas dataframe `data`, with each sample represented by the `text` column and its corresponding noisy label represented by the `label` column,  here is how you use `CrossValCurate` - 


```python linenums="1"

from dqc import CrossValCurate

cvc = CrossValCurate()
data_curated = cvc.fit_transform(data[['text', 'label']])
```
The result stored in `data_curated` which is a pandas dataframe similar to `data` with the following columns -
```python
>>> data_curated.columns
['text', 'label', 'label_correctness_score', 'is_label_correct', 'predicted_label', 'prediction_probability']
```

* `'label_correctness_score'` represents a normalized score quantifying the correctness of `'label'`. 
* `'is_label_correct'` is a boolean flag indicating whether the given `'label'` is correct (`True`) or incorrect (`False`). 
* `'predicted_label'` and `'prediction_probability'` represent the curation model's prediction and the corresponding probability score. 
 
For more details regarding different hyperparameters available in `CrossValCurate`, please refer to the [API documentation](https://sumanthprabhu.github.io/DQC-Toolkit/).
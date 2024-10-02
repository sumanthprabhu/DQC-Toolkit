<a href="https://github.com/sumanthprabhu/DQC-Toolkit/actions" alt="Build Status"><img src='https://img.shields.io/github/actions/workflow/status/sumanthprabhu/DQC-Toolkit/test.yml' alt="Build status"/></a> 
<a href="https://sumanthprabhu.github.io/DQC-Toolkit/latest/" alt="Docs Status"><img src='https://img.shields.io/website?url=https%3A%2F%2Fsumanthprabhu.github.io%2FDQC-Toolkit%2F&label=docs' alt="Docs status"/></a>
<a href='https://pypi.org/project/dqc-toolkit/'><img src='https://img.shields.io/pypi/pyversions/DQC-Toolkit' alt="Python version"/></a>
<a href='https://pypi.org/project/dqc-toolkit/'><img src='https://img.shields.io/pypi/v/DQC-Toolkit' alt='Pypi version' /></a> 
<a href='https://github.com/sumanthprabhu/DQC-Toolkit/blob/main/LICENSE'><img src='https://img.shields.io/pypi/l/DQC-toolkit' alt='Licence' /></a>

![](/docs/images/dqc-toolkit.svg)


DQC Toolkit is a Python library and framework designed with the goal to facilitate improvement of Machine Learning models by identifying and mitigating label errors in training dataset. Currently, DQC toolkit offers `CrossValCurate` and `LLMCurate`. `CrossValCurate` can be used for label error detection / correction in text classification (binary / multi-class) based on cross validation. `LLMCurate` extends [PEDAL: Enhancing Greedy Decoding with Large Language Models using Diverse Exemplars](https://arxiv.org/abs/2408.08869) to compute LLM-based confidence scores for free-text labels.

## Installation

Installation of DQC-toolkit can be done as shown below
```python
pip install dqc-toolkit
```

## Quick Start
### CrossValCurate
Assuming your text classification data is stored as a pandas dataframe `data`, with each sample represented by the `text` column and its corresponding noisy label represented by the `label` column,  here is how you use `CrossValCurate` - 


```python linenums="1"

from dqc import CrossValCurate

cvc = CrossValCurate()
data_curated = cvc.fit_transform(data[['text', 'label']])
```
The result stored in `data_curated` is a pandas dataframe similar to `data` with the following columns -
```python
>>> data_curated.columns
['text', 'label', 'label_correctness_score', 'is_label_correct', 'predicted_label', 'prediction_probability']
```

* `'label_correctness_score'` represents a normalized score quantifying the correctness of `'label'`. 
* `'is_label_correct'` is a boolean flag indicating whether the given `'label'` is correct (`True`) or incorrect (`False`). 
* `'predicted_label'` and `'prediction_probability'` represent the curation model's prediction and the corresponding probability score. 
 
### LLMCurate
Assuming `data` is a pandas dataframe containing samples with our target text for curation under column `column_to_curate`, here is how you use `LLMCurate` -
```python
    
    llmc = LLMCurate(model, tokenizer)
    ds = llmc.run(
            data,
            column_to_curate,
            ds_column_mapping,
            prompt_variants,
            llm_response_cleaned_column_list,
            answer_start_token,
            answer_end_token,
            batch_size,
            max_new_tokens
            )
```
where
* `model` and `tokenizer` are the instantiated LLM model and tokenizer objects respectively
* `ds_column_mapping` is the dictionary mapping of entities used in the LLM prompt to the corresponding columns in `data`. For example, `ds_column_mapping={'INPUT' : 'input_column'}` would imply that text under `input_column` in `data` would be passed to the LLM in the format `"[INPUT]row['input_column'][/INPUT]"` for each `row` in `data` 
* `prompt_variants` is the list of LLM prompts to be used to curate `column_to_curate` and `llm_response_cleaned_column_list` is the corresponding list of column names to store the reference responses generated using each prompt
* `answer_start_token` and `answer_end_token` are optional  text phrases representing the start and end of the answer respectively.

`ds` is a dataset object with the following additional features -

1. Feature for each column name in `llm_response_cleaned_column_list`
2. LLM Confidence score for each text in `column_to_curate`
 
For more details regarding different hyperparameters available in `CrossValCurate` and `LLMCurate`, please refer to the [API documentation](https://sumanthprabhu.github.io/DQC-Toolkit/).
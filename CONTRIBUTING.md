# Contribute to DQC Toolkit

Thank you for considering contributing to DQC Toolkit. 
Whether they are new features, bug fixes or better documentation , your contributions are valued ! 

PS - It would help us if you could give us a shout out in your social media posts whenever you find DQC Toolkit to be useful. Good old-fashioned repository ⭐️s are also welcome !

## Ways to contribute

All contributions are equally valuable. Following are some of the ways you can contribute to DQC Toolkit-

* Submitting bug reports or feature requests.
* Contributing code
* Improving documentation.

## Submitting bug reports or feature requests 

We use GitHub issues to track all bugs and feature requests. Before submitting, please make sure to verify that your issue is not already being addressed by existing issues or pull requests (PR).

### Submitting bug reports
When submitting an issue, please make sure to follow these guidelines - 
* Include a reproducible code snippet. 
* If an exception is raised, include the full traceback.
* Include your **OS type**, **OS version** and the version details for **Python**, **transformers**, **sentence-transformers** and **scikit-learn**. This information can be obtained by running the following code snippet:
```python 
>>> import dqc
>>> dqc.show_versions()
```

### Submitting feature requests
If there is a new feature you'd like to see in DQC Toolkit, please follow these guidelines when submitting the issue - 
* Describe what the feature is and the motivation behind it.
* Add code snippets (atleast pseudo code) to demonstrate how the feature will be used.
* Add links to any relevant literature (paper, blog, etc.) that can give more information about the proposed feature.

## Contributing code
To avoid duplicating work, it is recommended that you first search through the existing issues and PRs.
Following are the steps to follow when contributing code to DQC Toolkit - 
1. Submit an issue describing what you'd like to contribute. See [Submitting bug reports](#submitting-bug-reports) and [Submitting feature requests](#submitting-feature-requests). Please ensure to have an agreement with the maintainer, [@SumanthPrabhu](https://github.com/sumanthprabhu), before you start working on the issue.
2. Fork the repository. See [Forking a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository)
3. Sync your fork with the upstream repository. See [Sync your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#configuring-git-to-sync-your-fork-with-the-upstream-repository)
4. Create a new branch to hold your development changes.
5. Setup a virtual environment and install the development dependencies by running the following command - 
```bash
pip install -e ".[dev]"
``` 
You could also set it up by running -  
```bash
make install-dev
```
6. Develop your features. DQC Toolkit relies on docstrings for most of the documentation. Please ensure to write meaningful docstrings for your Class/ Method/ Function definitions. Refer the repository code for examples. DQC toolkit uses `ruff` to maintain code consistency. Run the following command after you've made your changes - 
```bash
make quality
```
To check the updated documentation, run the following - 
```bash
make docs
``` 

If you are adding new features, please make sure to add the corresponding tests. Also, ensure that your changes pass the existing tests. You can run the tests as shown below - 
```bash 
make test
```

7. Push your changes to your branch. Navigate to your fork of the repository and click on 'Pull Request' to open a pull request.

### Pull request checklist
- [ ] Provide a meaningful title to pull request that summarizes the contribution.
- [ ] If there are multiple issues that get resolved with your pull request, then add the corresponding issue numbers in the description to ensure they are linked.
- [ ] Until your pull requests are ready for review, please prefix the title with [WIP] to indicate a work in progress.
- [ ] Verify that code quality is maintained, relevant tests are added and all tests pass. 

## Improving documentation
 If you find that there is room for improvement in any form, you can submit an issue highlighting the suggested improvements and contribute as described in [Contributing code](#contributing-code) 

## Recognition
You will be given credit in the changelog if your contribution is part of the new release!
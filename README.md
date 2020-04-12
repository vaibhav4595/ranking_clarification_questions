### Assignment 3: State-of-the-art Reimplementation
Implementation of the paper [Learning to Ask Good Questions: Ranking Clarification Questions using Neural Expected Value of Perfect Information](https://arxiv.org/pdf/1805.04655.pdf) in Pytorch

Checkpoint 2 will involve reproducing the evaluation numbers of a state-of-the-art baseline model for the task of interest with code that you have implemented from scratch. In other words, you must get the same numbers as the previous paper on the same dataset.

In your report, also perform an analysis of what remaining errors this model makes (ideally with concrete examples of failure cases), and describe how you plan to create a new model for the final project that will address these error cases. If you are interested in tackling a task that does not have a neural baseline in the final project, you may also describe how you adopted the existing model to this new task and perform your error analysis on the new task (although you must report results on the task that the state-of-the-art model was originally designed for).


### Implementation Steps

- [x] Data Loading
- [x] Model
- [x] Integration
- [x] Experiments
- [x] Error Analysis

### Replication Steps

1. Clone the Repository, Create the Environment: ``` conda env create -f environment.yml ```

2. Download Data: ``` wget https://www.dropbox.com/s/8uaqm1ymrh50yxf/clarification_questions_dataset.zip ```, ```unzip clarification_questions_dataset.zip```

3. For Each Model Directory, Run Train & Test: ``` bash run.sh experiment_name```


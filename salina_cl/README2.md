# Continual Reinforcement Learning library
Continual Reinforcement Learning framework based on [SaLinA](https://github.com/facebookresearch/salina) with multiple scenarios and methods already implemented.

## Installation
* Make sure you have [pytorch installed with cuda>11.0](https://pytorch.org/). 
* Follow [instructions to install SaLinA](https://github.com/facebookresearch/salina#quick-start). 
* Install [jax with cuda support](https://github.com/google/jax#pip-installation-gpu-cuda).
* Then clone the repo and execute `pip install -e .` We recommend you to use Cuda 11.0 version (run `module load cuda/11.0`). 

## Getting started
Simply run the file `crl/run.py` with the desired config available in [configs](crl/configs/). You can select one of them with the flag `-cn=...`. Different scenarios are available in [configs/scenario](configs/scenario/). Simply add `scenario=...` as an argument. For example if you want to run our method CSP on the forgetting scenario of halfcheetah:
 ```console
python -m crl.run -cn=csp scenario=halfcheetah/forgetting
```

## Available methods

We implmented 8 different methods all built on top of soft-actor critic algorithm. To try them, just add the flag `-cn=my_method` on the command line. You can find the hps in [configs](crl/configs):

* `csp`: our method Continual Subspace of Policies

* `ft_1`: Fine-tune a single policy during the whole training
* `sac_n`: Fine-tune and save the policy at the end of the task. Start wit  a randomized policy when encountering a new task.
* `ft_n`: Fine-tune and save the policy at the end of the task. Clone the last policy when encountering a new task.
* `ft_l2`: Fine-tune a single policy during the whole training with a regularization cost (a simpler EWC method)
* `ewc`: see the paper [Overcoming catastrophic forgetting in neural](https://arxiv.org/pdf/1612.00796.pdf)
* `packnet`: see the paper [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/pdf/1711.05769.pdf)
* `pnn`: see the paper [Progressive Neural Networks](https://arxiv.org/pdf/1606.04671.pdf)



## Available scenarios

We propose 9 scenarios over 3 different [brax](https://github.com/google/brax) domains. To try them, just add the flag `scenario=...` on the command line:

* Halfcheetah:
    * `halfcheetah/forgetting`: 8 tasks - 1M samples for each task
    * `halfcheetah/transfer`: 8 tasks - 1M samples for each task
    * `halfcheetah/distraction`: 8 tasks - 1M samples for each task
    * `halfcheetah/composability`: 8 tasks - 1M samples for each task

* Ant:
    * `ant/forgetting`: 8 tasks - 1M samples for each task
    * `ant/transfer`: 8 tasks - 1M samples for each task
    * `ant/distraction`: 8 tasks - 1M samples for each task
    * `ant/composability`: 8 tasks - 1M samples for each task

* Humanoid:
    * `humanoid/hard`: 4 tasks - 2M samples for each task

## Organization of the repo

The `core.py` file contains the building blocks of this framework. Each experiment consists in running a `Framework` over a `Scenario`, i.e. a sequence of train and test `Task`. The models are learning procedures that use salina agents to interact with the tasks and learn from them through one or multiple algorithms.

* [frameworks](crl/frameworks/) contains generic learning procedures (e.g. using only one algorithm, or adding a regularization method in the end)
* [scenarios](crl/scenarios/) contains CRL scenarios i.e sequence of train and test tasks
* [algorithms](crl/algorithms/) contains different RL / CL algorithms (ppo, sac, td3, ewc regularization)
* [agents](crl/agents/) contains salina agents (policy, critic, ...)
* [configs](crl/configs/) contains the configs files of benchmarked methods/scenarios.

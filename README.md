# X-evolve: Solution space evolution powered by large language models

This repo is based on [DeepMind’s FunSearch](https://github.com/google-deepmind/funsearch) and the modified version by [RayZhhh](https://github.com/RayZhhh/funsearch).

There are three directories:

- `bin_packing` contains the dataset for the bin packing problem as well as the Python files required to run experiments on this problem.
- `cycle_graphs` contains the dataset for the Shannon capacity of cycle graphs problem.
- `implementation` contains an implenmentation of the X-evolve framework. It includes the concrete implementations of LLM call, Program database, and X-search algorithm. Moreover, both the LLM call and the Program database update processes are implemented in a multithreaded manner to improve execution efficiency.

## Installation

Before running our framework X-evolve, please ensure that the following Python libraries are installed in your Python environment: `requests`, `concurrent.futures`, `numpy`, `scikit-learn`, `logging`, and `re`. Additionally, make sure that your Python version is 3.9 or higher, as some of the libraries used in this framework are available in Python 3.x and are not supported in earlier versions.

## Usage

Please open the notebooks [admissible_set.ipynb](admissible_set.ipynb) and [cyclic_graphs.ipynb](cyclic_graphs.ipynb) to validate the new discoveries we found.

## How to run this project

1. Ensure that your Python environment satisfies the requirements specified in the Installation section.
2. Refer to the `config.py` to review the parameter settings for the specific problem you intend to run. You need to specify the target problem by passing the parameter `CONFIG_TYPE` via the command line. For the cap set problem, you are required to specify the dimension by passing the parameter `N_DIM` via the command line. For the cycle graphs problem, you must provide the `NODES_DIM` parameter, such as `NODES_DIM=9_5`, which indicates that the target graph is the 5th power of the 9-cycle, i.e., $\mathcal{C}_{9}^{\boxtimes 5}$. For the symmetry admissible set problem, use the `N_W_DIM` parameter, for example, `N_W_DIM=21_15`, which corresponds to computing the admissible set $\mathcal{A}(21,15)$. You can specify the location and naming of the generated log files by passing the `LOG_DIR` parameter.
3. Set your own large language model (LLM) API key and related invocation parameters in the [sample_llm_api.py](implementation/sample_llm_api.py).
4. We provide a simple script file [run.sh](run.sh) to execute the project. For example, to run the cycle graphs problem in $\mathcal{C}_{9}^{\boxtimes 5}$, simply execute the following command in the terminal: `CONFIG_TYPE=cycle_graphs NODES_DIM=9_5 LOG_DIR=log1 bash run.sh`.

## License

X-evolve is licensed under the [Apache-2.0](LICENSE) license.

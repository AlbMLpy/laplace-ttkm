Laplace Approximation For Tensor Train Kernel Machines In System Identification
=====

## ✨ Project Description
To address the scalability limitations of Gaussian process (GP) regression, several approximation techniques have been proposed. One such method is based on tensor networks, which utilizes an exponential number of basis functions without incurring exponential computational cost. However, extending this model to a fully probabilistic formulation introduces several design challenges. In particular, for tensor train (TT) models, it is unclear which TT-core should be treated in a Bayesian manner.

We introduce **a Bayesian tensor train kernel machine** that applies **Laplace approximation** to estimate the posterior distribution over a selected TT-core and employs **variational inference (VI)** for precision hyperparameters. 
Experiments show that core selection is largely independent of TT-ranks and feature structure, and that VI replaces cross-validation while offering up to 65× faster training. The method’s effectiveness is demonstrated on an inverse dynamics problem.

## 📊 Datasets
For ablation studies and real-data experiments, we use the following 7 UCI regression datasets (Dua and Graff, 2017) and industrial robot dataset (Weigand et al., 2022):

|  | Boston | Concrete | Energy | Kin8nm | Naval | Protein | Yacht | Robot Dynamics |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------| -------------| -------------|
| **N** | 506 | 1030 | 768 | 8192 | 11934 | 45730 | 308 | 43624 |
| **D** | 13 | 8 | 8 | 8 | 16 | 9 | 6 | 18 |

where **N** represents the number of samples and **D** denotes the data dimensionality.

## ⚙️ Environment
We use `conda` package manager to install required python packages. Once `conda is installed`, **run** the following command (while in the root of the repository):
```shell
conda env create -f environment.yml
```
This will create a new environment named `bayes_env` with all required packages already installed. You can **install** additional packages by running:
```shell
conda install <package name>
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `conda install jupyterlab` or with `conda install jupyter notebook`

## 🐳 (Optional) Running Experiments with Docker

Instead of setting up a Conda environment manually, you can run the entire experiment inside a Docker container. This ensures full reproducibility and requires only `Docker installed` on your system. 

ℹ️ Note: Depending on your Docker installation, you may need to prefix
all `docker` commands in this guide with `sudo`. 

1. From the project root (where the `Dockerfile` is located) **build** the Docker image:
    ```shell
    docker build -t la-ttkm-project .
    ```

2. **Run** the container interactively:
    ```shell
    docker run -it -v $(pwd)/experiments:/app/experiments --name la-ttkm la-ttkm-project
    ```
3. **You are all set to reproduce the Numerical Experiments!** 🤗

4. **Re-enter** the same container:
    ```shell
    docker start -ai la-ttkm
    ```

5. **Cleaning up** (optional):
    
    1. Remove the container:
        ```shell
        docker rm la-ttkm
        ```
    2. Remove the image:
        ```shell
        docker rmi la-ttkm-project
        ```

## 🚀 How to Reproduce the Numerical Experiments

0. **Activate** the virtual environment:
    ```shell
    conda activate bayes_env
    ```

1. **Run:**
   ```shell
   python setup_project.py
   ```
   to download all datasets and configure the project directories. 

2. Once the setup script has completed, **run:**
    ```shell
    cd experiments
    ```
    This folder contains three subdirectories, each corresponding to a distinct experiment described in the paper: `ablation_study`, `variational_inference` and `robot_dynamics`.

3. `ablation_study`
    -  **Run:** `cd ablation_study`
    -  **Analyze:** If `Docker` run `python analysis.py`. Otherwise, run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate figures stored in `artifacts`.

4. `variational_inference`
    -  **Run:** `cd variational_inference`
    -  **Train**: 
        ```bash
        python training.py 'all'
        ```
        Computes evaluation metrics for further comparison. These are stored in `artifacts/training_artifacts`. Use `python training.py --help` to see all options (e.g., parallel/sequential mode and `n_jobs`).
    -  **Analyze:** If `Docker` run `python analysis.py`. Otherwise, run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate the final table for comparison.

5. `robot_dynamics`:
    -  **Run:** `cd robot_dynamics`
    -  **Train:** 
        ```bash
        python training.py 'all'
        ```
        Computes evaluation metrics and predictions for further comparison. These are stored in `artifacts/training_artifacts`. 
    -  **Analyze:** If `Docker` run `python analysis.py`. Otherwise, run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate the final table for comparison and generate figures stored in `artifacts`.


## 📜 Citation

If you find our work helpful, please consider citing the paper:

```bibtex
In Progress!
```

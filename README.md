In Progress...
=====

## ✨ Project Description
In Progress...

## 📊 Datasets
In Progress...

## ⚙️ Environment
We use `conda` package manager to install required python packages. Once `conda is installed`, **run** the following command (while in the root of the repository):
```shell
conda env create -f environment.yml
```
This will create a new environment named `opt_env` with all required packages already installed. You can **install** additional packages by running:
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
    docker build -t opt-project .
    ```

2. **Run** the container interactively:
    ```shell
    docker run -it -v $(pwd)/experiments:/app/experiments --name opt opt-project
    ```
3. **You are all set to reproduce the Numerical Experiments!** 🤗

4. **Re-enter** the same container:
    ```shell
    docker start -ai opt
    ```

5. **Cleaning up** (optional):
    
    1. Remove the container:
        ```shell
        docker rm opt
        ```
    2. Remove the image:
        ```shell
        docker rmi opt-project
        ```

## 🚀 How to Reproduce the Numerical Experiments

0. **Activate** the virtual environment:
    ```shell
    conda activate opt_env
    ```

1. **Run:**
   ```shell
   python setup_project.py
   ```
   to download all datasets and configure the project directories. 

2. Once the setup script has completed, **run:**
    In Progress ...


## 📜 Citation

If you find our work helpful, please consider citing the paper:

```bibtex
In Progress!
```

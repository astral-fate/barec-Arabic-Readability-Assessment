D:
cd D:\arabic_readability_project


conda activate barec_env


conda install -c conda-forge notebook

jupyter notebook
__

setting env 

This log shows that your `barec_env` environment is in a broken state and is missing many essential libraries required by both Jupyter Notebook and the `camel-tools` package. Installing packages one by one has led to this incomplete setup.

The only reliable way to fix this is to delete the broken environment and create a new, clean one with a single command that installs everything at once.

-----

### Final Solution: Start Fresh

Please follow these steps exactly in your Anaconda Prompt. This will create a complete and stable environment from scratch.

#### **Step 1: Deactivate and Delete the Broken Environment**

```bash
conda deactivate
conda env remove --name barec_env
```

(Confirm with `y` if it asks.)

-----

#### **Step 2: Create and Activate a New, Clean Environment**

```bash
conda create --name barec_env python=3.10
y
```

-----

#### **Step 3: Install All Required Libraries with One Command**

This single, comprehensive command will install everything you need for the script and for Jupyter, with the correct versions pinned to prevent conflicts.

```bash
pip install notebook pandas scikit-learn accelerate>=0.21.0 camel-tools==1.5.6 transformers==4.43.4 numpy==1.26.4 emoji dill requests jinja2 packaging markupsafe python-dateutil
```

-----

#### **Step 4: Install GPU-Enabled PyTorch**

This command installs the correct version of PyTorch for your NVIDIA GPU.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


pip install transformers[torch]

or
pip install accelerate -U
-----

#### **Step 5: Download CAMeL Tools Data**

```bash
python -m camel_tools.cli.camel_data -i light
```
conda install -c conda-forge ipywidgets


### Next Steps

Your environment is now completely and correctly set up. To run your notebook:

1.  Make sure your `(barec_env)` is active.
2.  Set the isolation variable to prevent future conflicts:
    ```bash
    set PYTHONNOUSERSITE=1
    ```
3.  Launch Jupyter Notebook from the same prompt:
    ```bash
    jupyter notebook
    ```


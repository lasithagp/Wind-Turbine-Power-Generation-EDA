# 1. Create a new Conda environment with Python 3.11
conda create -n proj_windTurb python=3.11 -y

# 2. Activate the environment
conda activate proj_windTurb

# 3. Install required packages
conda install -y pandas numpy matplotlib seaborn scikit-learn statsmodels

# 4. Install Jupyter support so that this env can be used as a kernel
conda install -y ipykernel
python -m ipykernel install --user --name proj_windTurb --display-name "Python (proj_windTurb)"

# 5. Export environment to YAML (for GitHub)
conda env export --from-history > environment.yml

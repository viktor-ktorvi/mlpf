# MLPF - Machine Learning for Power Flow

## Installation

```
git clone https://github.com/viktor-ktorvi/mlpf.git
cd mlpf

conda create -n mlpfenv python=3.10
conda activate mlpfenv
pip install -r requirements.txt

```

## Generate docs

```
mkdir docs
cd docs

sphinx-quickstart
# fill out form
cd ..
sphinx-apidoc -o docs mlpf

cd docs
```

In _index.rst_ write 'modules' under ':caption: Contents:' like so:

```
   :caption: Contents:

   modules
```

In _conf.py_ change the following:

```python
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

...

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

...

html_theme = 'sphinx_rtd_theme'
```

Generate the html file:
```
.\make.bat html
```

I had to delete a bunch of unnecessary crap manually to make it look decent. There must be a better way.
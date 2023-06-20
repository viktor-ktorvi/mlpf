from pathlib import Path

from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='mlpf',
    version='0.0.6',
    description='Machine learning for power flow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author='Viktor Todosijevic',
    author_email='todosijevicviktor998@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords=['machine learning', 'power'],
    packages=find_packages(),
    install_requires=['numpy', 'pandapower', 'PYPOWER', 'scikit-learn', 'scipy', 'tqdm']
)

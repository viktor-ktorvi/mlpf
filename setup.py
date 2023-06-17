from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
]

setup(
    name='mlpf',
    version='0.0.1',
    description='Machine learning for power flow',
    url='',
    author='Viktor Todosijevic',
    author_email='todosijevicviktor998@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords=['machine learning', 'power'],
    packages=find_packages(),
    install_requires=['numpy', 'pandapower', 'PYPOWER', 'scikit-learn', 'scipy', 'tqdm', 'torchmetrics']
)

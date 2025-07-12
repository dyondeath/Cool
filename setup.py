from setuptools import setup, find_packages

setup(
    name='quantum_echo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'qutip',
        'numpy',
        'matplotlib',
        'statsmodels',
        'scipy',
        'pymc>=5.0',
        'arviz',
        'rich',
        'streamlit',
    ],
    description='Quantum Eraser Echo Simulator',
    author='[Your Name]',
    license='MIT',
)

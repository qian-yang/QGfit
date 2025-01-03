from setuptools import setup, find_packages

# Read requirements.txt for install_requires
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='qgfit',
    version='1.0.0',
    author='Qian Yang',
    author_email='qianyang.astro@gmail.com',
    description='A Python pipeline for quasar and galaxy spectral fitting.',
    packages=find_packages(),
    include_package_data=True,  # Ensure data files are included
    package_data={'qgfit': ['template/*']},  # Specify which files to include
    install_requires=requirements,  # Load from requirements.txt
    python_requires='>=3.8',
)


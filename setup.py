## initialization file
from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='SC2 Training CNN',
      author='Abhilash Dhal',
      author_email='adhal@ucdavis.edu',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'
      ],
      zip_safe=False)

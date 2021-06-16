#!/usr/bin/env python

from setuptools import setup, Extension
 

setup(name='Armageddon', 
      version='1.0', 
      description='Asteroid atmospheric entry solver', 
      author='AMCG project', 
      packages=['armageddon'], 
      package_dir = {'armageddon': 'armageddon'}, 
      package_data = {'armageddon':['resources/*.csv']}, 
      include_package_data=True 
     ) 

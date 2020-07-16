# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 01:08:04 2020

@author: joaovitor
"""

from setuptools import setup

setup(name='Synther',
      version='0.1.0b',
      description='Simple command line sound synthesizer.',
      url='https://github.com/Chum4k3r/synther.git',
      author='João Vitor Gutkoski Paes',
      author_email='joaovitorgpaes@gmail.com',
      license='MIT',
      install_requires=['numpy', 'numba', 'sounddevice', 'pynput'],
      packages=['synther'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"
          ])

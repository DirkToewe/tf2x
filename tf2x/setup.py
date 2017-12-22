from distutils.core import setup

setup(
  name='tf2x',
  version='0.1.0',
  description='Unofficial Tensorflow traversal and transformation utilities.',
  url='https://github.com/DirkToewe/tf2x',
  license='GPLv3',
  long_description=open('README.md').read(),
  install_requires = ['tensorflow', 'graphviz'],
  package_dir = {'': './src/main/python'},
  packages=[
    'tf2x'
  ],
  package_data={
    'tf2x': ['*.js', '*.template']
  }
)
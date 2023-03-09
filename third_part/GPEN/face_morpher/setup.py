from setuptools import setup, find_packages

# To test locally: python setup.py sdist bdist_wheel
# To upload to pypi: twine upload dist/*

setup(
  name='facemorpher',
  version='5.2.dev0',
  author='Alyssa Quek',
  author_email='alyssaquek@gmail.com',
  description=('Warp, morph and average human faces!'),
  keywords='face morphing, averaging, warping',
  url='https://github.com/alyssaq/face_morpher',
  license='MIT',
  packages=find_packages(),
  install_requires=[
    'docopt',
    'numpy',
    'scipy',
    'matplotlib',
    'dlib'
  ],
  entry_points={'console_scripts': [
      'facemorpher=facemorpher.morpher:main',
      'faceaverager=facemorpher.averager:main'
    ]
  },
  data_files=[('readme', ['README.rst'])],
  long_description=open('README.rst').read(),
)

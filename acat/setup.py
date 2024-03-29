from setuptools import setup

setup(
    name='acat',
    version='0.0.1',    
    description='Abelian Complexity Analysis Tool',
    url='https://github.com/paoloearth/acat',
    author='Paolo Pulcini',
    author_email='paoloearth@gmail.com',    
    license='MIT',
    packages=['acat'],
    install_requires=['pandas',
                      'numpy',
                      'matplotlib.pyplot'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',   
    ],
)    

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dsplib',
    version='1.0.0',
    author='Zach Beever',
    author_email='zbeever@bu.edu',
    description='Small DSP library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url= 'https://github.com/zbeever/dsplib',
    requires= ['numpy', 'tqdm', 'numba'],
    license= 'MIT',
    keywords= ['digital signal processing','dsp','wavelet','fourier','fft','stft','dwt','dtwt','hilbert'],
    packages= setuptools.find_packages(),
    package_data={'':['*.txt','*.md']},
    classifiers= [
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.6',
)

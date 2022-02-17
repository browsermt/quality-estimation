import setuptools


setuptools.setup(
    name='quality_estimation',
    description='Scripts for training feature-based quality estimation model',
    author='Marina Fomicheva',
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False
)

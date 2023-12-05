from setuptools import setup, find_packages

package_name = 'pcd_completion'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[],
    install_requires=[],
    zip_safe=True,
    maintainer='Saber Mohammadi',
    maintainer_email='saber.mohammadi1986@gmail.com',
    description='This package contains functions useful for ML algorithm use. It creates simple interface between some used AI algorithms.',
    license='GNU GENERAL PUBLIC LICENSE v3',
    entry_points={},
)

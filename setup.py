from setuptools import setup, find_packages

setup(
    name="y-scramble",
    version='0.0.3',
    packages=find_packages(),
    author="Frederico Schmitt Kremer",
    author_email="fred.s.kremer@gmail.com",
    description="Y-Scramble: a package for y-randomization validation",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    keywords="machine learnings qsar data science",
    install_requires = [
        "numpy", "scipy", "scikit-learn", "pytest"
    ]
)
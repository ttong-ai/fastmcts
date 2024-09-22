from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.21,<2.0"]

setup(
    author="Tony Tong",
    description="Fast Python implementation of Monte Carlo Tree Search with parallel capabilities",
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="mcts monte carlo tree search parallel fast",
    name="fastmcts",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "scripts"]),
    extras_require={
        "dev": ["pytest", "black"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    url="https://github.com/ttong-ai/fastmcts",  # Updated URL
    version="0.3.0",
)

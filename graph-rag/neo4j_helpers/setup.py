from setuptools import setup, find_packages

setup(
    name='neo4j_helpers',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'neo4j',
        'python-dotenv'
    ],
    author='Egor Safronov',
    author_email='egormsafronov@gmail.com',
    description='A package for basic Neo4j database operations',
)

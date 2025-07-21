# Known reaction network

Creates known enzymatic reaction network from enzyme-reaction pairs

## Getting Started

We recommend using uv to manage dependencies and set up virtual environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv python install 3.13 # Install uv
git clone git@github.com:stefanpate/krxns.git
cd krxns
uv sync
```

Create a file conf/filepaths/filepaths.yaml based on the structure in TEMPLATE_FILEPATHS_CONF.yaml

## Usage

### Convert atom mapped reactions into mass contributions format

Atom mapped reactions and associated compounds are extracted from [Rhea](https://www.rhea-db.org/) and [UniProt](https://www.uniprot.org/) using into a format outline in [this repository](https://github.com/stefanpate/enz-rxn-data). The output mass contributions format contains normalized mass contributions for reactant to product.

```
python scripts/process_reactions.py
```

A csv with compounds that counts the number of reactions they appear in is also created. Some of these compounds are designated as default sources of mass for pathfinding in the reaction network later on. One can change the default sources by editing the config file directly or using the notebook decide_sources.ipynb.

###
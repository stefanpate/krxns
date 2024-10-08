from krxns.cheminfo import expand_unpaired_cofactors
from krxns.config import filepaths
import pandas as pd
import json

unpaired_fp = filepaths['cofactors'] / "unpaired_cofactors_reference.tsv"

name_blacklist = [
    'acetyl-CoA',
]

unpaired_ref = pd.read_csv(
    filepath_or_buffer=unpaired_fp,
    sep='\t'
)

filtered_unpaired = unpaired_ref.loc[~unpaired_ref['Name'].isin(name_blacklist), :]
cofactors = expand_unpaired_cofactors(filtered_unpaired, k=10)

with open(filepaths["cofactors"] / "manually_added_unpaired.json", 'r') as f:
    manual = json.load(f)

cofactors = {**cofactors, ** manual}

with open(filepaths["cofactors"] / "241008_unpaired_cofactors.json", 'w') as f:
    json.dump(cofactors, f)
from string import digits
from itertools import chain

def str2int(stringy_dict: dict):
    '''
    Casts string keys to ints in multi-level, JSON-derived dicts
    '''
    if len(stringy_dict) == 0:
        return stringy_dict
    
    if type(next(iter(stringy_dict.values()))) is dict:
        stringy_dict = stringy_dict.copy()
        for k, v in stringy_dict.items():
            stringy_dict[k] = str2int(v)
    
    if all([char in digits for char in chain(*stringy_dict.keys())]):
        inty_dict = {int(k): v for k, v in stringy_dict.items()}
        return inty_dict
    else:
        return stringy_dict
    
if __name__ == '__main__':
    import json
    from krxns.config import filepaths
    foo = {'foo': 0, 'bar': 1}
    bar = {'0': 'foo', '1': 'bar'}
    baz = {'foo': bar, 'bar': bar}
    qux = {'0': baz, '1': baz}
    print(foo, "-->", str2int(foo))
    print(bar, "-->", str2int(bar))
    print(baz, "-->", str2int(baz))
    print(qux, "-->", str2int(qux))

    # Load sim connected reactions
    with open(filepaths['connected_reactions'] / 'sprhea_240310_v3_mapped_similarity.json', 'r') as f:
        sim_cxn = str2int(json.load(f))
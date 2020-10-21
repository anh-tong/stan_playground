import os
import pystan
import pickle
import hashlib

def create_model(model_code, cache=True, model_name=None, save_dir="./cache", **kwargs) -> pystan.StanModel:
    if not cache:
        return pystan.StanModel(model_code=model_code)
    else:
        hash = hashlib.md5(model_code.encode('ascii')).hexdigest()
        if model_name is None:
            cache_fn = f'cached_model_{hash}.pkl'
        else:
            cache_fn = f'cached_model_{model_name}_{hash}.pkl'

        cache_file = os.path.join(save_dir, cache_fn)

        try:
            sm = pickle.load(open(cache_file, 'rb'))
        except:
            sm = pystan.StanModel(model_code=model_code)
            with open(cache_file, 'wb') as f:
                pickle.dump(sm, f)
        else:
            print('Load from cached!')

        return sm

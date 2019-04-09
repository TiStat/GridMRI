import itertools
import warnings

from ModelScripts.U_gridexec_helpers import *

root = 'C:/Users/timru/Documents/CODE/deepMRI1/'


# (Carthesian Grid Configs) ----------------------------------------------------
# also consider, that filters must be of uneven length, if Unet!


for name in ['MSE', 'SSIM', 'MS-SSIM', 'MS-SSIM-GL1']:
    configs = {
        'lossflavour': [name],  # ['MSE', 'SSIM', ...],
        'batch': [2, 4],
        'kernel': [5, 9],
        'filters': [[125, 100, 50, 100, 125], [100, 100, 100, 100, 100],
                    [266, 266, 266, 266, 266], [314, 314, 314, 314, 314, 314, 314],
                    [100, 200, 400, 800, 400, 200, 100]],
        'reps': [[3, 3, 3, 3, 3], [2, 2, 2, 2, 2, 2, 2]],
        'dncnn_skip': [False],
        'sampling': [True],
        'Uskip': [True],
        'trainsteps': [40000]
    }


    # Cartesian product
    l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))

    # check for user error
    fil = [f['filters'] for f in l if len(f['filters'])%2 == 0]
    rep = [f['reps'] for f in l if len(f['reps'])%2 == 0]
    if len(fil) != 0 or len(rep)!= 0:
        warnings.warn('There is an even length Unet, look at filters :{} , reps:{}'.format( fil, rep))

    # remove invalid: mismatch of reps & filter lengths
    l = [d for d in l if len(d['filters']) == len(d['reps'])]
    with open(root + "{}.txt".format(name), "wb") as fp:  # Pickling
        pickle.dump(l, fp)

# test configs
configs = {
    'lossflavour': ['MSE'],
    'batch': [2],
    'kernel': [5],
    'filters': [[1,1,1], [1,1,1,1,1]],
    'reps': [ [1,1,1,1,1], [2,2,2]],
    'dncnn_skip': [False],
    'sampling': [True],
    'Uskip': [True],
    'trainsteps': [3]
}

# Cartesian product, but only valid architechtures; i.e. filters & reps match
l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
l = [d for d in l if len(d['filters'])==len(d['reps'])]

# make sure the .txt has same name as main script!
with open(root + "ModelScripts/URefactored.txt", "wb") as fp:  # Pickling
    pickle.dump(l, fp)



# Gather all the results
pandasResults(root = root, parentdir= 'Umodels', melt=True)


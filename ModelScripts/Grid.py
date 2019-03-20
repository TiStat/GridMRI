import itertools

configs = {
    'lossflavour': ['MAE','MSE', 'SSIM', 'MS-SSIM'], # 'MAE','MS-SSIM-GL1'],
    'batch': [2, 6],
    'kernel': [5, 9],
    'filters':[[50,100,200,100,50], [125,100,50,100,125], [100,100,100,100,100]], # , [100,100,100,100,100,100]],
    #[166, 166, 166],[200,100, 200], [100,300,100]
    # note that these are used for each op in rep
    'reps': [[3,3,3,3,3]],
    'dncnn_skip' : [False],
    # 'learning_rate'
    # 'optimizer' : ['ADAM', 'SGD']
    'trainsteps': [4000]
}

configs = {
    'lossflavour': ['MAE','MSE', 'SSIM', 'MS-SSIM'], # 'MAE','MS-SSIM-GL1'],
    'batch': [2, 6],
    'kernel': [5, 9],
    'filters':[[100,100,100,100,100,100]],
    #[166, 166, 166],[200,100, 200], [100,300,100]
    # note that these are used for each op in rep
    'reps': [[3,3,3,3,3]],
    'dncnn_skip' : [False],
    # 'learning_rate'
    # 'optimizer' : ['ADAM', 'SGD']
    'trainsteps': [4000]
}


#test config
configs = {
    'lossflavour' : ['MAE'],
    'batch' :[2,4],
    'kernel' : [5,7],
    'filters' : [[2, 4, 2], [2,5,2]],
    'reps': [[3,2,3]],
    'dncnn_skip': [False],
    'trainsteps': [10]
}


# (3) dict of configs
l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
#l.append({'runcounter':0})

# pickle the objects
import pickle
with open("grid.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)


# --------------------------------------------------------
with open("grid.txt", "rb") as fp:   # Unpickling
    grid = pickle.load(fp)
grid

for config in grid:
    kernel = config['kernel']
    filters = config['filters']
    print(kernel, filters)










# failing attempt: lists are stored as strings and must be obtained as literal
# l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
# grid = pd.DataFrame(l)
# grid['modelID'], grid['MAE'], grid['MSE'], grid['SSIM'], grid['trainruntime'] = [np.nan]*5
# grid.to_csv('C:/Users/timru/Documents/CODE/deepMRI1/modelHPC/grid.csv', index = False)

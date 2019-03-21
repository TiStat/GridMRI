import itertools
root = 'C:/Users/timru/Documents/CODE/deepMRI1/'
# be carefull to make filters and reps of same length
configs = {
    'lossflavour': ['MSE', 'SSIM'], # 'MAE','MS-SSIM-GL1'],
    'batch': [2, 6],
    'kernel': [5, 9],
    'filters':[[125,100,50,100,125], [100,100,100,100,100]], # , [100,100,100,100,100,100]],
    #[166, 166, 166],[200,100, 200], [100,300,100]
    # note that these are used for each op in rep
    'reps': [[3,3,3,3,3]],
    'dncnn_skip' : [False],
    # 'learning_rate'
    # 'optimizer' : ['ADAM', 'SGD']
    'trainsteps': [4000, 8000]
}

configs2 = {
    'lossflavour': ['MAE','MSE', 'SSIM'],
    'batch': [2, 6],
    'kernel': [5, 9],
    'filters':[[100,100,100,100,100,100,100], [100,200,400,800,400,200,100]],
    #[166, 166, 166],[200,100, 200], [100,300,100]
    # note that these are used for each op in rep
    'reps': [[3,3,3,3,3,3,3]],
    'dncnn_skip' : [False],
    # 'learning_rate'
    # 'optimizer' : ['ADAM', 'SGD']
    'trainsteps': [4000]
}

configs3 = {
    'lossflavour': ['MAE','MSE', 'SSIM', 'MS-SSIM'], # 'MAE','MS-SSIM-GL1'],
    'batch': [2, 6],
    'kernel': [5, 9],
    'filters':[[100,100,100,100,100,100]],
    #[166, 166, 166],[200,100, 200], [100,300,100]
    # note that these are used for each op in rep
    'reps': [[3,3,3,3,3,3]],
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
    'filters' : [[2, 4, 2]],
    'reps': [[3,2,3]],
    'dncnn_skip': [False],
    'sampling' : [True],
    'Uskip' : [True],
    'trainsteps': [10]
}


# (3) dict of configs
l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))

#l.append({'runcounter':0})

# pickle the objects
import pickle
with open(root + "/grid.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)


# (multiple depths):
l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
l2 = list(dict(zip(configs2, x)) for x in itertools.product(*configs2.values()))
l.append(l2)

import pickle
with open("grid.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)


# --------------------------------------------------------
with open("grid.txt", "rb") as fp:   # Unpickling
    grid = pickle.load(fp)

with open(root+"/grid/gridResult.txt", "rb") as fp:  # Unpickling
    gridres = pickle.load(fp)







# failing attempt: lists are stored as strings and must be obtained as literal
# l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
# grid = pd.DataFrame(l)
# grid['modelID'], grid['MAE'], grid['MSE'], grid['SSIM'], grid['trainruntime'] = [np.nan]*5
# grid.to_csv('C:/Users/timru/Documents/CODE/deepMRI1/modelHPC/grid.csv', index = False)

# (plotting) -------------------------------------------------------

# plotting all predictions' evals
import matplotlib.pyplot as plt
import numpy as np
with open("benchmark.txt", "rb") as fp:  # Unpickling
    bench = pickle.load(fp)

fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(18,6))
for name, group in bench.groupby('name'):
    print(name, group)
    group.plot(y='MAE', ax=ax1, label=name)
    group.plot(y='MSE', ax=ax2, label=name)
    group.plot(y='SSIM', ax=ax3, label=name)
line_labels = sorted(list(np.unique(bench['name'].values)), key = str.lower)
ax1.legend().set_visible(False)
ax2.legend().set_visible(False)
ax3.legend().set_visible(False)
fig.legend((ax1,ax2,ax3), labels=line_labels,   # The labels for each line
           loc="center right")
plt.show()


# plotting the mean statistics
import pandas as pd
with open("grid.txt", "rb") as fp:  # Unpickling
    g = pickle.load(fp)
g = pd.DataFrame(g)


ax.g.plot(x="name", y=["MAE", "MSE", "SSIM"], subplots=True, layout=(1,3))
plt.show


# proof of concept

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

y = np.random.rand(10,4)
y[:,0]= np.arange(10)
df = pd.DataFrame(y, columns=["name", "MAE", "MSE", "SSIM"])

ax = df.plot(x="name", y=["MAE", "MSE", "SSIM"], kind="bar", subplots=True, layout=(1,3))
plt.show()


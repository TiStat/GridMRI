import itertools
import pickle
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

root = 'C:/Users/timru/Documents/CODE/deepMRI1/'


# (Carthesian Grid Configs) ----------------------------------------------------
# be carefull to make  all filters and reps of same length, for all instances!
# otherwise due to carthesian, dim mismatch.
# also consider, that filters must be of uneven length, if Unet!

# TODO options to be implemented
# 'learning_rate'
# 'optimizer' : ['ADAM', 'SGD']

# levels: 3, filter: 500, total filters: 1500 (reps=3) -- except for last filters version: comparative in total filters

for name in ['MSE', 'SSIM', 'MS-SSIM', 'MS-SSIM-GL1']:
    configs = {
        'lossflavour': [name],  # 'MSE', 'SSIM'],
        'batch': [2, 4],
        'kernel': [5, 9],
        'filters': [[125, 100, 50, 100, 125], [100, 100, 100, 100, 100], [266, 266, 266, 266, 266]],
        'reps': [[3, 3, 3, 3, 3]],
        # 'filters':[[314,314,314,314,314,314,314], [100,200,400,800,400,200,100]],
        # 'reps': [[2,2,2,2,2,2,2]],
        'dncnn_skip': [False],
        'sampling': [True],
        'Uskip': [True],
        'trainsteps': [4000, 6000]
    }

    l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
    with open(root + "finalsmall_{}.txt".format(name), "wb") as fp:  # Pickling
        pickle.dump(l, fp)

# test configs
configs = {
    'lossflavour': ['MSE'],
    'batch': [2],
    'kernel': [5],
    'filters': [[1,1,1]],
    'reps': [[1,1,1]],
    # 'filters':[[314,314,314,314,314,314,314], [100,200,400,800,400,200,100]],
    # 'reps': [[2,2,2,2,2,2,2]],
    'dncnn_skip': [False],
    'sampling': [True],
    'Uskip': [True],
    'trainsteps': [3]
}

l = list(dict(zip(configs, x)) for x in itertools.product(*configs.values()))
with open(root + "U.txt", "wb") as fp:  # Pickling
    pickle.dump(l, fp)


def Unet_model_fn (features, labels, mode, lossflavour, reps, filters, kernel, batch, Uskip, sampling, dncnn_skip):


# (concat Results pd) ----------------------------------------
filenames = glob.glob(root + 'HPC results save/Results/GridResult stopped/*')
GRID = list()
for gridres in filenames:
    with open(gridres, "rb") as fp:  # Unpickling
        grid = pickle.load(fp)

        GRID.extend(grid)
Gb = pd.DataFrame(GRID)  # with benchmark
Gb.filters = Gb.filters.astype(str)
Gb.reps = Gb.reps.astype(str)

b = Gb[Gb['name'] == 'benchmark'].iloc[0].iloc[0:3]  # mean benchmark

Gb = Gb[Gb.name != 'benchmark']
Gb = Gb.dropna(subset = ['name'])
Gb = Gb.drop_duplicates(keep = 'first')

Gb.traintime.sum()

Gb.iloc[9]
Gb = Gb.drop(39)
# with outliers: Gb,  without:
meltedg = pd.melt(Gb, id_vars=['name',  'lossflavour', 'kernel', 'reps', 'filters', 'batch', 'trainsteps'],
                  value_vars=['M-MAE', 'M-MSE', 'M-SSIM'])

meltedg.to_csv(root + 'HPC results save/Results/gridResult.csv')

Gb.to_csv(root + 'HPC results save/Results/gridR.csv')

Gb[Gb.name == '4_Unet_MS-SSIM-GL1_3_28_5_0_30'][['M-MAE', 'M-MSE', 'M-SSIM']]
Gb.groupby('lossflavour')['lossflavour'].count()
# (concat benchmark pd) -------------------------------------
filenames = glob.glob(root + 'HPC results save/Results/Bench stopped/*')
BENCH = pd.DataFrame()
for file in filenames:
    with open(file, "rb") as fp:  # Unpickling
        b = pickle.load(fp)

        bench = pd.DataFrame(b[0])
        for i in range(1, len(b)):
            bench = bench.append(pd.DataFrame(b[i]))  # single img bench
        BENCH = BENCH.append(bench)

BENCH.to_csv(root + 'HPC results save/Results/BENCH.csv')
meltedBENCH = pd.melt(BENCH, id_vars=['name'],
                  value_vars=['MAE', 'MSE', 'SSIM'])
meltedBENCH.to_csv(root + 'HPC results save/Results/meltedBENCH.csv')

# aggregated stat
BENCH.groupby('name')['MSE'].mean()
BENCH.groupby('name')['MAE'].mean()
BENCH.groupby('name')['SSIM'].mean()

# (plot BENCH) -----------------------------------------------
#ind = random.sample(list(range(300)), k=50)
#ind = sorted(ind)
#bench = bench[bench.index.isin(ind)]

#ind = [str(i) for i in ind]
#ind = pd.DataFrame(ind)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for name, group in bench.groupby('name'):
    group.plot(y='MAE', ax=ax1, label=name)
    group.plot(y='MSE', ax=ax2, label=name)
    group.plot(y='SSIM', ax=ax3, label=name)
    l = len(group)
# plotting all 300: use these ticks
plt.setp((ax1, ax2, ax3), xticks=np.arange(0, l, 20.0))
#ax1.set_xticklabels(labels = ind)
#ax2.set_xticklabels(labels = ind)
#ax3.set_xticklabels(labels = ind)
ax1.legend().set_visible(False)
ax2.legend().set_visible(False)
ax3.legend().set_visible(False)
ax1.set_title('MAE')
ax2.set_title('MSE')
ax3.set_title('SSIM')

line_labels = sorted(list(np.unique(bench['name'].values)), key=str.lower)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()



# --------------------------------------------------------
with open("grid.txt", "rb") as fp:  # Unpickling
    grid = pickle.load(fp)
pd.DataFrame(grid)

with open(root + 'modelHPC/' + "3_23_2_28_9_grid2img/gridResult.txt", "rb") as fp:  # Unpickling
    gridres = pickle.load(fp)
grid = pd.DataFrame(gridres)

with open('C:/Users/timru/Documents/CODE/deepMRI1/modelHPC/3_22_23_48_8_grid2img/benchmark.txt',
          "rb") as fp:  # Unpickling
    b = pickle.load(fp)
b = pd.DataFrame(b)

pd.read_pickle('C:/Users/timru/Documents/CODE/deepMRI1/modelHPC/3_22_23_48_8_grid2img/benchmark.txt')
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

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for name, group in bench.groupby('name'):
    print(name, group)
    group.plot(y='MAE', ax=ax1, label=name)
    group.plot(y='MSE', ax=ax2, label=name)
    group.plot(y='SSIM', ax=ax3, label=name)
line_labels = sorted(list(np.unique(bench['name'].values)), key=str.lower)
ax1.legend().set_visible(False)
ax2.legend().set_visible(False)
ax3.legend().set_visible(False)
fig.legend((ax1, ax2, ax3), labels=line_labels,  # The labels for each line
           loc="center right")
plt.show()

# plotting the mean statistics -----------------------
import pandas as pd

with open('modelHPC' + "/3_22_13_45_43_grid2/gridResult.txt", "rb") as fp:  # Unpickling
    g = pickle.load(fp)
g = pd.DataFrame(g)

g = g[['name', "M-MAE", "M-MSE", "M-SSIM"]]
ax = g.plot(y=["M-MAE", "M-MSE", "M-SSIM"], kind='bar', subplots=True, layout=(1, 3))
plt.show()

# plot benchmark images ---------------------------
with open("grid0/benchmark.txt", "rb") as fp:  # Unpickling
    bench = pickle.load(fp)
bench = pd.DataFrame(bench)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for name, group in bench.groupby('name'):
    # print(name, group)
    group.plot(y='MAE', ax=ax1, label=name)
    group.plot(y='MSE', ax=ax2, label=name)
    group.plot(y='SSIM', ax=ax3, label=name)
    # l = len(group)
    # plt.xticks(np.arange(0, l, 1.0))
    l = len(group)

for ax, name in zip((ax1, ax2, ax3), ('MAE', 'MSE', 'SSIM')):
    ax.legend().set_visible(False)
    ax.set_title(str(name))

line_labels = sorted(list(np.unique(bench['name'].values)), key=str.lower)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.setp((ax1, ax2, ax3), xticks=np.arange(0, l, 1.0))

plt.show()

# proof of concept
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

y = np.random.rand(10, 4)
y[:, 0] = np.arange(10)
df = pd.DataFrame(y, columns=["name", "MAE", "MSE", "SSIM"])

ax = df.plot(y=["MAE", "MSE", "SSIM"], kind="bar", subplots=True, layout=(1, 3))
plt.show()

directory = 'C:/Users/timru/Desktop/HPC results save/Results'



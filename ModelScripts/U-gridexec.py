# grid search related
import os
import pickle
import imageio

# image saving & subset test_data related
from datetime import datetime

# script imports:
from ModelScripts.U_model_fn import *
from ModelScripts.U_gridexec_helpers import *


# (system configuration) -------------------------------------------------------
def Gridit (platform=['hpc', 'cloud', 'home'][2],
            gridname='{0}.txt'.format(os.path.basename(__file__)[:-3]),
            init=True, gridparent='Umodels',
            lineartraversal=True,
            saveimage=True):
    '''
    :gridname: gridname.txt is grid to be executed. __file__ must be gridname.py
        both must be in root dir.
    :init: boolean specifying, if script runs for the first time or is supposed to
    start off from gridfolder and get on with the due work
    :gridfolder: foldername in root, where 'gridname' shall be continued
    :lineartraversal: boolean: if init = False, there are two ways to continue the grid:
        a) True: start at the grid index, where it left off.
        b) False: gather all not yet run parameter grids (including the one where
           it stopped) and run all of these.
    :saveimage: write out the predicted images of all models + once true and noisy
    '''
    root = {
        'hpc': '/scratch2/truhkop/',
        'cloud': '/home/cloud/',
        'home': 'C:/Users/timru/Documents/CODE/deepMRI1/'}[platform]

    # (load data) --------------------------------------------------------------
    train_data, train_labels, test_data, test_labels = loadData(platform, root)

    # seed for printing images
    random.seed(42)
    imgind = random.sample(list(range(test_data.shape[0])), k=10)

    # (Overhead for first & later runs) ----------------------------------------
    gridfolder = gridname[:-4]
    gridparentpath = root + gridparent + '/'
    gridfolderpath =  gridparentpath + gridfolder + '/'
    gridimgpath = gridparentpath + gridfolder + 'img/'

    if init:
        # (mkgrid folder) -------------------------------
        os.mkdir(gridfolderpath)

        # (initialize checkpoints) --------------------
        with open(gridname, "rb") as fp:  # unpickle
            grid = pickle.load(fp)

        with open(gridparentpath + "Result_{}.txt".format(gridname), "wb") as fp:  # pickling
            res = grid.copy()
            pickle.dump(res, fp)

        # (initialize benchmark file) -----------------
        with open(gridparentpath + "benchmark_{}.txt".format(gridname), "wb") as fp:  # pickling
            print('Benchmarking...')
            MAE, MSE, SSIM = benchit(test_data, test_labels)
            pickle.dump(
                [{'name': 'benchmark',
              'MAE': MAE,
              'MSE': MSE,
              'SSIM': SSIM}], fp)
            print('Benchmarked')

        # (save benchmark img pairs) ------------------
        if saveimage:
            os.mkdir(gridimgpath)
            for ind in imgind:
                noisy = (test_data[ind] * 255).astype(np.uint8)
                trueim = (test_labels[ind] * 255).astype(np.uint8)
                imageio.imwrite(gridimgpath + 'noisy_{}.jpg'.format(ind), noisy)
                imageio.imwrite(gridimgpath + 'trueim_{}.jpg'.format(ind), trueim)


    elif not init:
        # (set how to proceed grid) ----------------
        if not (gridname[:-4] in gridfolder):
            raise Exception('Carefull, you are trying to overwrite {} \n'
                            'with grid: {}'.format(gridfolder[:-1], gridname))

        with open(gridname, "rb") as fp:
            grid = pickle.load(fp)

        with open(gridparentpath + "Result_{}.txt".format(gridname), 'rb') as fp:  # Unpickling
            gridResult = pickle.load(fp)

        if lineartraversal:
            # figure out where in grid to start the run, if all previous models actually ran:
            dictlengths = [len(gridResult[i]) for i in range(len(gridResult))]
            if len(grid[0]) + 5 in dictlengths:
                # if models were successful, 5 additional keys are added to that grid in gridResult
                indexlastfinished = len(dictlengths) - 1 - dictlengths[::-1].index(len(grid[0]))
                indexunfinished = (indexlastfinished + 1)

                # new grid of unfinished
                grid = grid[indexunfinished:]

        elif not lineartraversal:
            # figure out which configs to run, if some had been skipped (gridResult
            # values are exacly as long as those of grid
            grid = [grid[i] for i, v in enumerate(gridResult) if len(v) == len(grid[0])]
            c = [i for i, v in enumerate(gridResult) if len(v) == len(grid[0])]

    # (GRID RUN) ---------------------------------------------------------------
    # setting up the parameter environment, upon which model_fn harvests
    for counter, config in enumerate(grid):
        lossflavour = config['lossflavour']
        trainsteps = config['trainsteps']
        batch = config['batch']

        # ensure gridresult is filled properly and
        # does not overwrite already successful models
        if init:
            c = counter
        elif lineartraversal:
            c = counter + indexunfinished
        elif not lineartraversal:
            c = c[counter]  # this is only a hint!

        # (set up ESTIMATORS) --------------------------------------------------
        d = datetime.now()
        modelID = '{}_Unet_{}_{}_{}_{}_{}_{}'.format(
        c, lossflavour, d.month, d.day, d.hour, d.minute, d.second)

        Unet = tf.estimator.Estimator(
            model_fn=Unet_model_fn,
            model_dir=gridfolderpath + modelID,
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                log_step_count_steps=1000),
            params={'kernel': config['kernel'],
                    'filters': config['filters'],
                    'lossflavour': config['lossflavour'],
                    'reps': config['reps'],
                    'dncnn_skip': config['dncnn_skip'],
                    'sampling': config['sampling'],
                    'Uskip': config['Uskip']}
    )

        print('-------------- generated {} Estimator  ------------'.format(modelID))
        print('Config: {}'.format(config))

        # (TRAINING) -----------------------------------------------------------
        start_time = datetime.now()
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_labels,
            batch_size=batch,
            num_epochs=None,
            shuffle=True)
        try:
            Unet.train(input_fn=train_input_fn, steps=trainsteps)
        except Exception as e:
            print(e)
            continue

        # write out to grid.txt note the view!
        time_elapsed = datetime.now() - start_time
        print('######### Trained {} ########'.format(modelID))
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

        # (EVALUATE) -----------------------------------------------------------
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=test_data,
            y=test_labels,
            batch_size=1,
            num_epochs=1,
            shuffle=False)

        # evaluation for each prediction stored in benchmark.txt
        try:
            predictions = list(Unet.predict(input_fn=test_input_fn))
        except Exception as d:
            print(d)
            continue

        print('######### predicted {} #########'.format(modelID))

        # (CHECKPOINTS) --------------------------------------------------------
        # (Checkpoint Benchmark after model) ---------------
        with open(gridparentpath + "benchmark_{}.txt".format(gridname), "rb") as fp:  # Unpickling
            MAE, MSE, SSIM = benchit(predictions, test_labels)
            bench = pickle.load(fp)
            b = bench.copy()

        with open(gridparentpath + "benchmark_{}.txt".format(gridname), "wb") as fp:  # pickling
            b.append({'name': modelID, 'MAE': MAE, 'MSE': MSE, 'SSIM': SSIM})
            pickle.dump(b, fp)

        # (Checkpoint gridResult after model) --------------
        with open(gridparentpath + "Result_{}.txt".format(gridname), "rb") as fp:  # Unpickling
            gridResult = pickle.load(fp)

        with open(gridparentpath + "Result_{}.txt".format(gridname), "wb") as fp:  # pickling
            # aggregate the results and write to gridResult.txt,
            # note that 'currentconfig' is only a view on gridResult

            currentconfig = gridResult[c]

            currentconfig['name'] = modelID
            currentconfig['traintime'] = time_elapsed
            currentconfig['M-MAE'] = np.mean(MAE)
            currentconfig['M-MSE'] = np.mean(MSE)
            currentconfig['M-SSIM'] = np.mean(SSIM)
            pickle.dump(gridResult, fp)

        print('######### Evaluated {} #########'.format(modelID))

        # (save selcted predictions) --------------
        if saveimage:
            for ind in imgind:
                img = (predictions[ind] * 255).astype(np.uint8)
                imageio.imwrite(gridimgpath + '{}_predicted_{}.jpg'.format(modelID, ind), img)

        print('DONE')


Gridit(platform=['hpc', 'cloud', 'home'][2],
       gridname='{0}.txt'.format(os.path.basename(__file__)[:-3]),
       init=True,
       lineartraversal=True,
       saveimage=True)

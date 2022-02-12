import os
import numpy as np
from raise_utils.interpret import DODGEInterpreter

kinds = ['FBC', 'FBG', 'FC', 'FG']
metrics = ['f1', 'acc', 'pd', 'pf', 'prec', 'auc']

for kind in kinds:
    print(kind)
    print('=' * len(kind))

    result = {x: [] for x in metrics}
    for file in os.listdir('./log/' + kind + '_train_sampled'):
        try:
            path = './log/' + kind + '_train_sampled/' + file
            interp = DODGEInterpreter(
                files=[path], n_iters=10, metrics=metrics)
            results = interp.interpret()[file]

            for metric in metrics:
                result[metric].append(results[metric][0])
        except:
            pass

    for k, v in result.items():
        print(k + ':', np.median(v))

    print()

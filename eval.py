import numpy as np
import pandas as pd

results = pd.read_csv('results.csv')

print('Mean Accuracy (Mean Hit Ratio): ', np.mean(results['acc']))

count = 0
for i in range(results.shape[0]):
    if results.iloc[i]['acc'] == 1:
        count += 1

print('Number of Puzzles Solved Completely Correctly: ', count)
print('Total Number of Puzzles: ', results.shape[0])
print('Number of Puzzles Solved Completely Correctly / Total Number of Puzzles', count/results.shape[0])
import sys
import cv2
import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import permutations
import time

def format_label(label: str) -> np.ndarray:
    """
    Takes the input label as string,
    returns the array version of it.
    """

    label_arr = []
    for i in range(0, len(label), 2):
        label_arr.append(int(label[i]))
    return np.array(label_arr)

def create_image(img: np.ndarray, permutation: np.ndarray, m = 3, n = 3) -> np.ndarray:
    """
    Takes the input image and the wanted permutation,
    returns the rearranged image as the permutation indicates.
    By default, takes 3-by-3 images in terms of number of puzzle pieces stated with m and n respectively.
    """

    rows, cols, channels = img.shape

    # Determining the row boundaries that seperate the image pieces
    row_boundaries = []
    for i in range(1, m+1):
        row_boundaries.append((i * rows) // m)

    # Determining the column boundaries that seperate the image pieces
    col_boundaries = []
    for i in range(1, n+1):
        col_boundaries.append((i * cols) // n)

    img_to_return = np.zeros_like(img)

    # Creating the new rearranged image
    for i in range(permutation.shape[0]):
        # Extract the wanted piece of the image
        piece_index = (i//n, i%n)
        prev_row_boundary = 0 if piece_index[0] == 0 else row_boundaries[piece_index[0]-1]
        prev_col_boundary = 0 if piece_index[1] == 0 else col_boundaries[piece_index[1]-1]
        piece = deepcopy(img[prev_row_boundary:row_boundaries[piece_index[0]], prev_col_boundary:col_boundaries[piece_index[1]], :])

        # Place the extracted piece to the wanted location
        piece_index = (permutation[i]//n, permutation[i]%n)
        prev_row_boundary = 0 if piece_index[0] == 0 else row_boundaries[piece_index[0]-1]
        prev_col_boundary = 0 if piece_index[1] == 0 else col_boundaries[piece_index[1]-1]
        img_to_return[prev_row_boundary:row_boundaries[piece_index[0]], prev_col_boundary:col_boundaries[piece_index[1]], :] = piece
    
    return img_to_return

def solve(img: np.ndarray, m = 3, n = 3):
    """
    Solves the problem with the approach which aims to find the permutation that
    minimizes the number of edge pixels in the boundary areas between the puzzle pieces.
    By default, works on 3-by-3 images in terms of number of puzzle pieces stated with m and n respectively.
    """

    all_permutations = np.array(list(permutations(np.array(range(0, m*n)))))

    min_summation = sys.maxsize
    minimizing_permutation = -1

    rows, cols, channels = img.shape

    # Determining the row boundaries that seperate the image pieces
    row_boundaries = []
    for i in range(1, m):
        row_boundaries.append((i * rows) // m)

    # Determining the column boundaries that seperate the image pieces
    col_boundaries = []
    for i in range(1, n):
        col_boundaries.append((i * cols) // n)

    for i in range(all_permutations.shape[0]):
        permuted_img = create_image(img, all_permutations[i])
        edge_img = cv2.Canny(permuted_img, 150, 225, L2gradient=True)
        
        summation = 0

        for q in range(m-1):
            summation += np.sum(edge_img[row_boundaries[q]-1:row_boundaries[q]+1, :])

        for q in range(n-1):
            summation += np.sum(edge_img[col_boundaries[q]-1:col_boundaries[q]+1, :])

        if summation < min_summation:
            min_summation = summation
            minimizing_permutation = i

    return all_permutations[minimizing_permutation], create_image(img, all_permutations[minimizing_permutation], m, n)


### DENEME
no = 10
path = 'data/valid/' + str(no) + '.jpg'
img = cv2.imread(path)
df = pd.read_csv('data/valid.csv')

start = time.time()

print('Correct Solution: ', df.iloc[no]['label'])
solving_permutation, solved_img = solve(img)
print('Solution Found: ', solving_permutation)

end = time.time()
print('Execution Time: ', str(end - start))

cv2.imshow('original', img)
cv2.imshow('solved', solved_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
### DENEME
import base64
import io
import json
from pprint import pprint

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# 0. # = black (usually the background)        #000000
# 1. B = blue                                  #0074D9
# 2. R = red                                   #FF4136
# 3. G = green                                 #2ECC40
# 4. Y = yellow                                #FFDC00
# 5. W = white                                 #FFFFFF
# 6. V = violet                                #8A2BE2
# 7. O = orange                                #FF851B
# 8. T = teal                                  #7FDBFF
# 9. P = pink                                  #FF10F0
colors = ['#', 'B', 'R', 'G', 'Y', 'W', 'V', 'O', 'T', 'P']

plt.ioff()


def data2grid(sample: list[list[int]]) -> list[str]:
    return [' ' + ' '.join([colors[col] for col in row]) for row in sample]


def grid2data(grid: list[str]) -> list[list[int]]:
    return [[colors.index(cell) for cell in row.strip().split()] for row in grid]


def plot_grid(grid: list[str], plot: bool = True):
    """ Plot a grid ('#', 'B', 'R', 'G', 'Y', 'W', 'V', 'O', 'T', 'P') using matplotlib """
    cmap = mcolors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#FFFFFF', '#8A2BE2', '#FF851B', '#7FDBFF', '#FF10F0',
    ])
    norm = mcolors.BoundaryNorm(range(11), cmap.N)
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [[colors.index(cell) for cell in row.split()] for row in grid]

    ax.imshow(data, cmap=cmap, norm=norm)
    # Adding grid lines
    ax.set_xticks([i - 0.5 for i in range(len(data[0]) + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(len(data) + 1)], minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    ax.set_xticks([])
    ax.set_yticks([])
    if plot:
        plt.show()

    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='jpg')
    io_bytes.seek(0)
    plt.close()
    return base64.b64encode(io_bytes.read()).decode()


if __name__ == '__main__':
    print(data2grid([[0, 1, 2], [3, 4, 5], [6, 7, 9]]))
    print(grid2data(data2grid([[0, 1, 2], [3, 4, 5], [6, 7, 9]])))
    plot_grid(data2grid([[0, 1, 2], [3, 4, 5], [6, 7, 9]]))
    len(plot_grid(data2grid([[0, 1, 2], [3, 4, 5], [6, 7, 9]]), plot=False))

    with open('arc/data/training/0a938d79.json') as f:
        data = json.load(f)
        print('Example Puzzles:')
        for i, samples in enumerate(data['train']):
            print(f'Example #{i}')
            print('Input:')
            pprint(data2grid(samples['input']))
            print('Output:')
            pprint(data2grid(samples['output']))
            print('\n')

            # plot_grid(data2grid(samples['input']))
            # plot_grid(data2grid(samples['output']))
            # break

        print('------------------------------------')
        print('Test:')
        for i, samples in enumerate(data['test']):
            pprint(data2grid(samples['input']))
            pprint(data2grid(samples['output']))

"""
The augmentation module is responsible for providing more data related to each example with several transformations.
Those transformations introduce some human inductive bias to help the model analyze the patterns better.
"""
import json
from collections import Counter

import numpy as np

from preprocess import data2grid


def search(small: np.ndarray, large: np.ndarray) -> list[tuple[int, int]] | tuple[int, int] | None:
    """
    Search for the small grid in the large grid.
    :param small: The small grid
    :param large: The large grid
    :return: The top-left coordinates (r, c) of the small grid in the large grid
    """
    if small.shape == large.shape:
        return None

    res = []
    for r in range(large.shape[0] - small.shape[0] + 1):
        for c in range(large.shape[1] - small.shape[1] + 1):
            if np.all(large[r:r + small.shape[0], c:c + small.shape[1]] == small):
                res.append((r, c))
    return res if len(res) > 1 else res[0] if len(res) == 1 else None


def search_prompt(inp: np.ndarray, out: np.ndarray) -> str:
    """
    :param inp: input grid
    :param out: output grid
    :return: the prompt that represents the augmentations

    [Different sizes] `search(in, out)` or `search(out, in)` => tell coordinates + operation

    * search without modifications
    * search with rotations
    * search with flips
    * search with transpose
    """
    # Exact search
    prompt = ''
    res = search(inp, out)
    prompt += f'One can find the exact input grid in the output grid starting at the top-left coordinates {res}.\n' if res else ''
    res = search(out, inp)
    prompt += f'One can find the exact output grid in the input grid starting at the top-left coordinates {res}.\n' if res else ''

    if prompt != '':
        return prompt

    # Search with rotations
    for i in range(4):
        rotated = np.rot90(inp, i)
        res = search(rotated, out)
        prompt += f'One can find the rotated input grid in the output grid starting at the top-left coordinates {res} after rotating the input grid by 90 degrees {i} times.\n' if res else ''

        res = search(out, rotated)
        prompt += f'One can find the output grid in the rotated input grid starting at the top-left coordinates {res} after rotating the input grid by 90 degrees {i} times.\n' if res else ''

    # Search with horizontal flip
    res = search(np.fliplr(inp), out)
    prompt += f'One can find the horizontally flipped input grid in the output grid starting at the top-left coordinates {res}.\n' if res else ''
    res = search(np.fliplr(out), inp)
    prompt += f'One can find the horizontally flipped output grid in the input grid starting at the top-left coordinates {res}.\n' if res else ''

    # Search with vertical flip
    res = search(np.flipud(inp), out)
    prompt += f'One can find the vertically flipped input grid in the output grid starting at the top-left coordinates {res}.\n' if res else ''
    res = search(np.flipud(out), inp)
    prompt += f'One can find the vertically flipped output grid in the input grid starting at the top-left coordinates {res}.\n' if res else ''

    # Search with horizontal and vertical flip
    res = search(np.flip(inp), out)
    prompt += f'One can find the horizontally and vertically flipped input grid in the output grid starting at the top-left coordinates {res}.\n' if res else ''
    res = search(np.flip(out), inp)
    prompt += f'One can find the horizontally and vertically flipped output grid in the input grid starting at the top-left coordinates {res}.\n' if res else ''

    # Search with transpose
    res = search(np.transpose(inp), out)
    prompt += f'One can find the transposed input grid in the output grid starting at the top-left coordinates {res}.\n' if res else ''
    res = search(np.transpose(out), inp)
    prompt += f'One can find the transposed output grid in the input grid starting at the top-left coordinates {res}.\n' if res else ''

    return prompt


def components(grid: np.ndarray) -> np.ndarray:
    """
    :param grid: input grid
    :return: the connected components of the grid (each component is a unique integer)
    """
    res = np.zeros(grid.shape, dtype=int)
    def fill(r, c, color, component):
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1] or grid[r, c] != color or res[r, c] != 0:
            return
        res[r, c] = component
        fill(r - 1, c, color, component)
        fill(r + 1, c, color, component)
        fill(r, c - 1, color, component)
        fill(r, c + 1, color, component)

    component_id = 1
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if res[row, col] == 0:
                fill(row, col, grid[row, col], component_id)
                component_id += 1
    return res


def component_prompt(inp: np.ndarray, out: np.ndarray | None = None) -> str:
    """
    tell numbers per component + component size + color + first coordinate
    :param inp: input grid
    :param out: output grid
    :return: string representation of the connected components in both grids (only if the number of components is small)
    """
    in_components = components(inp)
    in_unique = np.unique(in_components)
    in_max_possible = in_components.shape[0] * in_components.shape[1]
    prompt = ''

    # Should be at most 30% of all possible to add to prompt
    if len(in_unique) < in_max_possible * 0.3 or len(in_unique) < 5:
        prompt += f'The input grid has {len(in_unique)} components. '
        prompt += 'Here is the input grid represented as components instead of colors. '
        prompt += 'Each component is represented with a unique integer:\n'
        prompt += '\n'.join([' ' + ' '.join(map(str, row)) for row in in_components]) + '\n'
        prompt += '\n'
        for i, component in enumerate(in_unique):
            if component == 0:
                continue
            component_size = np.sum(in_components == component)
            color = inp[in_components == component][0]
            r, c = np.where(in_components == component)
            prompt += f'{i + 1}. Component {component} has size {component_size}, color {color} and starts at top-left coordinates ({r[0]}, {c[0]}).\n'
        prompt += '\n'

    if out is None:
        return prompt

    out_components = components(out)
    out_unique = np.unique(out_components)
    out_max_possible = out_components.shape[0] * out_components.shape[1]

    if len(out_unique) < out_max_possible * 0.3 or len(out_unique) < 5:
        prompt += f'The output grid has {len(out_unique)} components. '
        prompt += 'Here is the output grid represented as components instead of colors. '
        prompt += 'Each component is represented with a unique integer:\n'
        prompt += '\n'.join([' ' + ' '.join(map(str, row)) for row in out_components]) + '\n'
        prompt += '\n'
        for i, component in enumerate(out_unique):
            if component == 0:
                continue
            component_size = np.sum(out_components == component)
            color = out[out_components == component][0]
            r, c = np.where(out_components == component)
            prompt += f'{i + 1}. Component {component} has size {component_size}, color {color} and starts at top-left coordinates ({r[0]}, {c[0]}).\n'
        prompt += '\n'
    return prompt


def diff_prompt(inp: np.ndarray, out: np.ndarray) -> str:
    """
    :param inp: input grid
    :param out: output grid
    :return: the prompt that represents the difference between input and output grids
    """
    if inp.shape != out.shape:
        return ''

    # Create a binary '!'/'.' grid where '!' means the cell is different and '.' means it is the same
    # Include the definition of the symbols in the prompt
    diff = np.where(inp != out, '!', '.')
    # If '!' is more than 30% => don't return anything
    if np.sum(diff == '!') > 0.3 * diff.size and np.sum(diff == '!') > 20:
        return ''

    prompt = 'The input grid differs from the output grid in the following cells '
    prompt += '("!" means that the cells are different, while "." means that they are the same):\n'
    prompt += '\n'.join([' ' + ' '.join(row) for row in diff]) + '\n'
    return prompt


def stats_prompt(inp: np.ndarray, out: np.ndarray | None = None) -> str:
    """
    :param inp: input grid
    :param out: output grid
    :return: the prompt that represents the statistics of the input and output grids
    """
    prompt = ''
    prompt += f'The input grid has {inp.shape[0]} rows, {inp.shape[1]} columns, and {len(np.unique(inp))} unique colors.\n'
    prompt += 'The counts for each color are:\n'
    for color, count in Counter(inp.flatten()).most_common():
        prompt += f'- {color}: {count}\n'

    if out is None:
        return prompt

    prompt += '\n'
    prompt += f'The output grid has {out.shape[0]} rows, {out.shape[1]} columns, and {len(np.unique(out))} unique colors.\n'
    prompt += 'The counts for each color are:\n'
    for color, count in Counter(out.flatten()).most_common():
        prompt += f'- {color}: {count}\n'

    return prompt


def get_train_augmentations(data: dict) -> str:
    prompt = ''
    for i, samples in enumerate(data['train']):
        prompt += f'\n### Example {i + 1}'
        inp = data2grid(samples['input'])
        out = data2grid(samples['output'])
        inp = np.array([list(row.replace(' ', '')) for row in inp])
        out = np.array([list(row.replace(' ', '')) for row in out])

        prompt += f'\n{stats_prompt(inp, out)}\n'

        res = search_prompt(inp, out)
        prompt += f'\n{res}\n' if res else ''

        res = component_prompt(inp, out)
        prompt += f'\n{res}\n' if res else ''

        res = diff_prompt(inp, out)
        prompt += f'\n{res}\n' if res else ''

    return prompt.strip()


def get_test_augmentations(data: dict) -> str:
    prompt = ''
    assert len(data['test']) == 1, 'There should be only a single test sample'
    for i, samples in enumerate(data['test']):
        inp = data2grid(samples['input'])
        inp = np.array([list(row.replace(' ', '')) for row in inp])

        prompt += f'\n{stats_prompt(inp)}\n'
        res = component_prompt(inp)
        prompt += f'\n{res}\n' if res else ''

    return prompt.strip()


if __name__ == '__main__':
    with open('arc/data/training/794b24be.json') as f:
        print(get_train_augmentations(json.load(f)))
    print('-' * 50)

    with open('arc/data/training/2dee498d.json') as f:
        data = json.load(f)
        print(get_train_augmentations(data))
        print('-' * 50)
        print('TEST PROMPT:')
        print(get_test_augmentations(data))
    print('-' * 50)

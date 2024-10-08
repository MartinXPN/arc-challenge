# ARC Challenge

## TODO
- [ ] Add Llama-based models
- [ ] Add example solutions to the prediction prompt?
- [ ] Add a validation + refinement step?

## Augmentations
We augment each example with several transformations that introduce some human inductive bias to help the model.

1. [Different sizes] + [Same sizes] `search(in, out)` or `search(out, in)` => tell coordinates + operation
   * search without modifications
   * search with rotations
   * search with flips (horizontal, vertical, horizontal + vertical)
   * search with transpose
   * search by ignoring the most common color

2. [Different sizes] + [Same sizes] components(in) => tell numbers per component + component size + color + first coordinate

   This should only be shown only if there are few components

3. [Same sizes] `diff(out, in): 1/0 matrix` => show what changed

    This should only be shown only if it's mostly cancelled out (1s are <30%)

4. [Diff sizes] + [Same sizes] `stats(in)` and `stats(out)`
   1. Count by color

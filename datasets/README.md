# Dataset structure — supported layouts

Two formats are supported:

A) Concatenated pairs (same as pix2pix style)
- train/
  - 0001.png  (left = input A, right = target B)
  - 0002.png
- test/
  - 1001.png

B) Separate folders
- train/
  - input/
    - 0001.png
    - 0002.png
  - target/
    - 0001.png
    - 0002.png
- test/
  - input/
  - target/

Important:
- If using separate folders, filenames must match between `input/` and `target/`.
- Images will be resized to `--load_size` (default 512) during training and testing.
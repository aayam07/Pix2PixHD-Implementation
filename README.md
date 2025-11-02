# Pix2PixHD Implementation

This repository contains a simplified but complete PyTorch implementation of Pix2PixHD.
It supports:
- Coarse-to-fine generator (global + local)
- Multi-scale discriminators
- Perceptual (VGG) loss
- Paired dataset formats: concatenated images or separate input/target folders

## Quick start

1. Create environment:
   - `python3 -m venv venv && source venv/bin/activate`
   - `pip install -r requirements.txt`

2. Add missing packages (edit torch version to match CUDA on your GPU if necessary).

3. Run a quick smoke test:
   - `python3 scripts/train.py --dataroot /path/to/dataset --name quick_test --batch_size 1 --load_size 256 --n_epochs 1`

4. Run test:
   - `python3 scripts/test.py --dataroot /path/to/dataset --name quick_test --which_epoch latest`

## Data formats supported

See `datasets/README.md` for dataset layout examples.

## Output

- Checkpoints saved to `./checkpoints/<name>/`
- Test results saved to `./results/<name>/` (images contain input | generated | target)

## Notes

- If you encounter issues, run the quick test and share `logs/quick_test.log`, checkpoint, and a sample output image.
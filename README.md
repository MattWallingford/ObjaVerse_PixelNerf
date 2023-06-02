# pixelNeRF adapted for ObjaVerse-XL

Adapted from PixelNerf for training on ObjaVerse-XL. Loader converts camera pose from Blender coordinates to OpenCV for matrix inversion, then converts to OpenGL to be consistent with PixelNerF. 

Includes gradient Clipping and epsilon trick to avoid exploding gradients when dividing by depth close to zero. 
Command to run: python3 train/train.py -n objaverse_Large -c conf/exp/ObjaVerse.conf -D ~/renders/ --gpu_id='0 1 2 3 4 5 6 7'  --epochs 1000 -B 32 --resume

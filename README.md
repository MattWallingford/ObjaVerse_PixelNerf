# pixelNeRF: Neural Radiance Fields from One or Few Images

Adapted from PixelNerf for training on ObjaVerse-XL. Loader converts camera pose from Blender coordinates to OpenCV for matrix inversion, then converts to OpenGL to be consistent with PixelNerF. 

Includes gradient Clipping and epsilon trick to avoid exploding gradients when dividing by depth close to zero. 

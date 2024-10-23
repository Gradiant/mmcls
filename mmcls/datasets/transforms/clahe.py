from mmcv.transforms import BaseTransform
from mmcls.datasets import TRANSFORMS
import numpy as np
import cv2


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    def __init__(self, clipLimit: float = 3.0, gridSize: tuple = (8, 8)) -> None:
        if isinstance(gridSize, tuple) and len(gridSize) == 2:
            self.gridSize = gridSize
        elif isinstance(gridSize, int):
            self.gridSize = (gridSize, gridSize)
        else:
            assert isinstance(gridSize, tuple), "gridsize should be a tuple of length 2 or an int"
        assert isinstance(clipLimit, float)
        self.clipLimit = clipLimit
        self.clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                     tileGridSize=self.gridSize)

    def transform(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            # print(type(img), img.dtype)
            # img = np.rint(img)
            # img = np.uint16(img)
            img[..., 0] = self.clahe.apply(img[..., 0])
            img[..., 1] = self.clahe.apply(img[..., 1])
            img[..., 2] = self.clahe.apply(img[..., 2])
            img_out = np.float32(img)
            results[key] = img_out
        return results

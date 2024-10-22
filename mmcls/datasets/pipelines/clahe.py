from mmcls.datasets import PIPELINES
import cv2
import numpy as np


@PIPELINES.register_module()
class CLAHE(object):
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

    def __call__(self, results):
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

    def __repr__(self):
        return self.__class__.__name__ + f'(clipLimit={self.clipLimit}, gridSize={self.gridSize})'

import os
import numpy as np
from PIL import Image

# ncempy 체크
try:
    import ncempy.io as nio
    DM3_AVAILABLE = True
except ImportError:
    DM3_AVAILABLE = False

def load_image_data(path):
    _, ext = os.path.splitext(path)
    if ext.lower() == '.dm3':
        if not DM3_AVAILABLE: 
            raise ImportError("ncempy 라이브러리가 필요합니다.")
        dm3 = nio.read(path)
        img_data = dm3['data']
        if img_data.ndim == 3: img_data = img_data[0, :, :]
        return img_data
    else:
        return np.array(Image.open(path).convert('L'))
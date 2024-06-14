from xarray.core import dataset
from shapely.geometry import shape


def extract_pixel_values(point, raster: dataset.Dataset):
    point = shape(['geometry'])
    x_coord, y_coord = point.x, point.y
    pixel_value = raster.sel(x=x_coord, y=y_coord, method='nearest')
    return pixel_value

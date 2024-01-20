import fiona
from fiona.io import MemoryFile
from shapely import buffer, centroid
from shapely.geometry import shape
from shapely import Point


def mem_buffer(shp_path: str, buffer_size: float = 1, driver: str = 'ESRI Shapefile'):

    with fiona.open(shp_path, 'r') as src:
        schema = src.schema.copy()

        mem_file = MemoryFile()
        with mem_file.open(driver=driver, schema=schema) as mem:
            for feature in src:
                geom = shape(feature['geometry'])
                neg_buffer = buffer(geom, buffer_size)
                mem.write({
                    'geometry': shape(neg_buffer),
                    'properties': feature['properties']
                })

    return mem_file


def mem_centroid(shp_path: str, driver: str = 'ESRI Shapefile'):

    with fiona.open(shp_path, 'r') as src:
        schema = src.schema.copy()
        schema['geometry'] = 'Point'

        mem_file = MemoryFile()
        with mem_file.open(driver=driver, schema=schema) as mem:
            for feature in src:
                geom = shape(feature['geometry'])
                centroid_geom = centroid(geom)
                mem.write({
                    'geometry': Point(centroid_geom),
                    'properties': feature['properties']
                })

    return mem_file

import ee

from ee import Image
from ee import ImageCollection
from ee import Geometry
from ee import Date
from ee import Filter

import json

ee.Initialize()


def does_intersect(image: Image, feature_geom: Geometry):
    """
    This function checks if both geoms intersects.
    Args:
        image: An ee.Image
        feature_geom: An ee.Geometry

    Returns:
        True or false.
    """
    intersects_ = image.geometry(maxError=0.001).intersects(feature_geom, maxError=0.001)

    return intersects_.getInfo()


def cloud_pixels(image: Image, cloud_percent: int, aoi: Geometry):
    cloud_band = image.select('MSK_CLDPRB').gt(cloud_percent)
    pixel_count = cloud_band.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=10,
        maxPixels=1e9
    )
    total_pixels = cloud_band.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi,
        scale=10
    )

    return tuple((pixel_count.get('MSK_CLDPRB').getInfo(), total_pixels.get('MSK_CLDPRB').getInfo()))


def export_images(folder: str, img: Image, aoi: Geometry, to_crs: str, filename: str):
    """
        Esta función exporta imágenes a un directorio en Google Drive.
    Args:
        folder: Directorio de destino en la raíz del drive
        img: Imagen de ee
        aoi: Región de interés
        to_crs: EPSG deseado para las imágenes exportadas
        filename: Nombre del archivo

    Returns:
        Una confirmación por cada imagen exportada correctamente.
    """
    raw_image = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])

    export_params = {
        'driveFolder': f'{folder}',
        'fileFormat': 'GeoTiff',
        'scale': 10,
        'region': aoi,
        'crs': to_crs
    }

    task = ee.batch.Export.image.toDrive(image=raw_image, description=filename, **export_params)

    task.start()
    task.status()

    return print(f"{filename} exportada correctamente.")


start_date = Date('2020-01-01')
end_date = Date('2023-12-31')

ic = ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start=start_date, opt_end=end_date).filter(
    Filter.eq('MGRS_TILE', '14QML')
)

print(ic.size().getInfo())

with open(r"C:\Users\Isai\Documents\Tesis\code\Parcelas\poligonos_parcelas.geojson", encoding='utf-8') as geo:
    parcels = json.load(geo)


inside = 0
out = 0
cloud_free = 0
semi_free = 0

for parcel in parcels['features']:
    parcel_feature = ee.Feature(parcel)
    parcel_bbox = parcel_feature.bounds()

    for image_ in ic.getInfo()['features']:
        image_ = Image(image_['id'])
        img_props = json.dumps(image_.getInfo(), indent=4)
        intersects = does_intersect(image=image_, feature_geom=parcel_bbox.geometry())
        if intersects:
            cloudy_pixels = cloud_pixels(image=image_, cloud_percent=50, aoi=parcel_bbox.geometry())
            inside += 1
            if '1' not in cloudy_pixels[0]:
                print(f"Image Id: {image_.id().getInfo()} limpia, {cloudy_pixels[0]}, Parcela: {parcel_feature.getInfo()['properties']['Id']}")
                export_images(folder='Tesis_cloudless', img=image_, aoi=parcel_bbox.geometry(), to_crs='EPSG:32614', filename=f"{parcel_feature.getInfo()['properties']['Id']}_{image_.id().getInfo()}")
                # print(f"{parcel_feature.getInfo()['properties']['Id']}_{image_.id().getInfo()}")
                with open(rf"G:/Mi unidad/Proyecto_tesis/metadata_cloudless/{parcel_feature.getInfo()['properties']['Id']}_{image_.id().getInfo()}.json", 'w') as f:
                    f.write(img_props)
                cloud_free += 1
            '''
            elif '0' in cloudy_pixels[0]:
                if '1' in cloudy_pixels[0]:
                    if cloudy_pixels[0]['0'] > cloudy_pixels[0]['1']:
                        print(f"Image Id: {image_.id().getInfo()} semi-limpia, {cloudy_pixels[0]}, Parcela: {parcel_feature.getInfo()['properties']['Id']}")
                        export_images(folder='Tesis_9', img=image_, aoi=parcel_bbox.geometry(), to_crs='EPSG:32614',
                                      filename=f"{parcel_feature.getInfo()['properties']['Id']}_{image_.id().getInfo()}")
                        with open(rf"G:/Mi unidad/Proyecto_tesis/metadata/{parcel_feature.getInfo()['properties']['Id']}_{image_.id().getInfo()}.json", 'w') as f:
                            f.write(img_props)

                        # print(f"{parcel_feature.getInfo()['properties']['Id']}_{image_.id().getInfo()} exportada")
                        semi_free += 1
            '''
        else:
            out += 1

print(f"Dentro: {inside}, Fuera: {out}, Libre de nubes: {cloud_free}, Semi-limpia: {semi_free}")

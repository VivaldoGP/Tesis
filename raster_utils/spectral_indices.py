from rasterio import DatasetReader


def ndvi(img: DatasetReader, nir_band: int = 8, red_band: int = 4):
    """
    Calcula el ndvi
    Args:
        nir_band:
        red_band:
        img: un imagen de rasterio

    Returns:
        un ndarray con los valores del ndvi para pixel
    """
    nir_band = img.read(nir_band)
    red_band = img.read(red_band)

    ndvi_band = (nir_band.astype(float) - red_band.astype(float)) / (nir_band.astype(float) + red_band.astype(float))

    return ndvi_band


def gndvi(img: DatasetReader, nir_band: int = 8, green_band: int = 3):

    nir_band = img.read(nir_band)
    green_band = img.read(green_band)

    gndvi_band = (nir_band.astype(float) - green_band.astype(float)) / (nir_band.astype(float) + green_band.astype(float))

    return gndvi_band


def evi(img: DatasetReader, nir_band: int = 8, red_band: int = 4, blue_band: int = 2,
        g: int = 2.5, c1: float = 6.0, c2: float = 7.5, l: float = 1.0):

    nir_band = img.read(nir_band)
    red_band = img.read(red_band)
    blue_band = img.read(blue_band)

    evi_band = g * ((nir_band.astype(float) - red_band.astype(float)) / (nir_band.astype(float) + c1 * red_band.astype(float) - c2 * blue_band.astype(float) + l))

    return evi_band


def msi(img: DatasetReader, swir_band: int = 11, nir_band: int = 8):
    """
    Calcula el msi
    Args:
        swir_band:
        nir_band:
        img: Una imagen de rasterio

    Returns:
        un ndarray con los valores del msi para pixel
    """
    swir_band = img.read(swir_band)
    nir_band = img.read(nir_band)

    msi_band = swir_band.astype(float) / nir_band.astype(float)

    return msi_band


def ndmi(img: DatasetReader, swir_band: int = 11, nir_band: int = 8):
    """
    Calcula el ndmi
    Args:
        swir_band:
        nir_band:
        img: un imagen de rasterio

    Returns:
        un ndarray con los valores del ndvi para pixel
    """
    nir_band = img.read(nir_band)
    swir_band = img.read(swir_band)

    ndmi_band = (nir_band.astype(float) - swir_band.astype(float)) / (nir_band.astype(float) + swir_band.astype(float))

    return ndmi_band


def ndwi(img: DatasetReader, nir_band: int = 8, green_band: int = 3):
    """
    Calcula el ndwi
    Args:
        nir_band:
        green_band:
        img: un imagen de rasterio

    Returns:
        un ndarray con los valores del ndwi para pixel
    """
    nir_band = img.read(nir_band)
    green_band = img.read(green_band)

    ndwi_band = (green_band.astype(float) - nir_band.astype(float)) / (green_band.astype(float) + nir_band.astype(float))

    return ndwi_band

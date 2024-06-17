import ee
ee.Authenticate()
ee.Initialize()
print(ee.String('Hello from the Earth Engine servers!').getInfo())

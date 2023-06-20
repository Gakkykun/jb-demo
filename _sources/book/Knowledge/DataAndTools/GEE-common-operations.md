# Google Earth Engine common operations
1. Getting coordinates of interest
2. Extracting the satellite data information from a specific point
3. Calibration/validation 
4. Error analysis
5. Creating a time-series
6. Water area delineation
7. Creating a mapping of a specific variable for the entire lake, or a portion of the lake in case the lake size is beyond the GEE computational capacity.

## Getting coordinates of interest
A coordinate of interest can be easily obtained at the Google Earth Engine Code Editor, where a pin can be put in the map that appears in the bottom half. A new variable will be created based on the pin, and the coordinate information can be extracted from the code editor.

## Extracting the satellite data information from a specific point
We first need to identify which dataset should be used.
We can extract reflectance spectra at a specific point, considering 3 × 3 neighborhood pixels centered at the point and take mean reflectance values for available wavelengths. 

## Calibration/validation 
A curve_fit function from Scipy can be used.

## Error analysis
There are different indices proposed as error metrics. They can be calculated without much efforts using Scikit-learn methods. Confusion matrix for classification problems can be also created using the library.

## Creating a time-series
Extracted information from satellite imagery can be converted into a Pandas dataframe, with which a time-series plotting can be done.

## Water area delineation
We can effectively obtain accurate water area using the method proposed by Donchyts et al, 2016 and the JWP database.

## Creating a mapping of a specific variable for the entire lake,  or a portion of the lake in case the lake size is beyond the GEE computational capacity.
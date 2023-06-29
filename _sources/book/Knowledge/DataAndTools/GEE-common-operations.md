---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Google Earth Engine common operations
1. Extracting the satellite data information from a specific point
2. Calibration/validation with error analysis
3. Creating a time-series
4. Creating a mapping of a specific variable.


## Extracting the satellite data information from a specific point
We first need to identify which dataset should be used.


```{code-cell} ipython3
startDate = '2019-01-01'
endDate = '2019-12-30'
sampling_point = ee.Geometry.Point([35.238484, -14.618457])

Sen2ImageCollection = ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(sampling_point)
        .filterDate(startDate, endDate)
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)
        .map(cloudMasking_S2_div)
        .map(rescaling)
```

We can extract reflectance spectra at a specific point, considering 3 × 3 neighborhood pixels centered at the point and take mean reflectance values for available wavelengths. 

```{code-cell} ipython3
Sen2Image = ee.ImageCollection('COPERNICUS/S2_SR')\
 .filterBounds(sampling_point)\
 .filterDate(startDate, endDate)\
 .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)\
 .map(cloudMasking)\
 .map(rescaling)\
 .median()\
 .divide(10000)

args = {
        'reducer': ee.Reducer.toList(),
        'geometry':sampling_point.buffer(30).bounds(),
        'scale': 20, 
        'maxPixels': 4e4,
        'bestEffort': True
    }

msi_info = Sen2Image.reduceRegion(**args).getInfo()

df = pd.DataFrame(msi_info)
df = df[['B1', 'B2','B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11']]
mean = df[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11']].mean()
```

## Calibration/validation with error analysis


```{code-cell} ipython3
Sen2ImageCollection = ee.ImageCollection('COPERNICUS/S2_SR')\
.filterBounds(sampling_point)\
.filterDate(startDate, endDate)\
.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)\
.select(S2_BANDS)\
.map(cloudMasking)

withChla = Sen2ImageCollection.map(addChla).select('Chla')
bounds_inlet = inlet.buffer(300)

reduce_chla = create_reduce_region_function(
    geometry=bounds_inlet, reducer=ee.Reducer.mean())

chla_stat_fc = ee.FeatureCollection(withChla.map(reduce_chla)).filter(
    ee.Filter.notNull(withChla.first().bandNames()))


chla_dict = fc_to_dict(chla_stat_fc).getInfo()
chla_df = pd.DataFrame(chla_dict)
```

Convert the time field into a datetime and keep the columns of interest.
```{code-cell} ipython3
chla_df['datetime'] = pd.to_datetime(chla_df['millis'], unit='ms')

chla_df = chla_df[['Chla','datetime']]

S_NDCI = np.concatenate((chla_df_w6['NDCI'].values,\
                         chla_df_o6['NDCI'].values,\
                         chla_df_m6['NDCI'].values,\
                         np.array([chla_df_i6['NDCI'].values.mean()]),\
                         chla_df_e6['NDCI'].values,\
                         chla_df_w9['NDCI'].values,\
                         chla_df_o9['NDCI'].values,\
                         chla_df_m9['NDCI'].values,\
                         np.array([chla_df_i9['NDCI'].values.mean()]),\
                         chla_df_e9['NDCI'].values\
                        ), axis=0) 
```

```{code-cell} ipython3
chla_jun = np.array([3.1128, 4.5416, 3.7401, 3.1390, 2.6231])
chla_sep = np.array([13.3576, 6.5588, 5.2463, 9.3652, 7.5128])

chla_insitu =np.concatenate((chla_jun, chla_sep), axis=0)
```

A curve_fit function from Numpy or Scipy can be used.

```{code-cell} ipython3
polynominal_coeff=np.polyfit(S_NDCI, chla_insitu, 2)


def NDCI_algorithm(x):
    return 9.547 + 104.809*x + 431.976*x**2

NDCI_v = np.vectorize(NDCI_algorithm)
Yhat = NDCI_v(S_NDCI)
Y = chla_insitu

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1-d1.dot(d1)/d2.dot(d2)

p = np.poly1d(np.polyfit(S_NDCI, chla_insitu,2))
t = np.linspace(-0.125, 0.05, 100)
plt.xlabel('Satellite derived NDCI')
plt.ylabel('In-situ Chal [$mg/m^3$]')
plt.title(f'$R^2$ = {round(r2, 2)}')
plt.plot(S_NDCI, chla_insitu, 'o', t, p(t), '-', color='k')
```

Validation
```{code-cell} ipython3
S_NDCI_val = np.concatenate((chla_df_w5['NDCI'].values,\
                            chla_df_o5['NDCI'].values,\
                            chla_df_m5['NDCI'].values,\
                             np.array([chla_df_i5['NDCI'].values.mean()]),\
                             chla_df_e5['NDCI'].values,\
                             chla_df_w7['NDCI'].values,\
                             chla_df_o7['NDCI'].values,\
                             chla_df_m7['NDCI'].values,\
                             chla_df_i7['NDCI'].values,\
                         chla_df_w8['NDCI'].values,\
                         chla_df_o8['NDCI'].values,\
                         chla_df_m8['NDCI'].values,\
                         np.array([chla_df_i8['NDCI'].values.mean()]),\
                         chla_df_e8['NDCI'].values,\
                         chla_df_w10['NDCI'].values,\
                         chla_df_o10['NDCI'].values,\
                         chla_df_m10['NDCI'].values,\
                         chla_df_i10['NDCI'].values,\
                         chla_df_e10['NDCI'].values\
                        ), axis=0) 

chla_may = np.array([3.9182, 3.6110, 2.0985, 2.0985, 3.2866])
chla_jul = np.array([3.2195, 4.6712, 3.4762, 4.2744])
chla_aug = np.array([11.1313, 6.2017, 8.9199, 3.1478, 8.7450])
chla_oct = np.array([12.2135, 9.4616, 4.5621, 13.3576, 6.7654])

chla_insitu_val =np.concatenate((chla_may, chla_jul, chla_aug, chla_oct), axis=0)

X_sat = NDCI_v(S_NDCI_val)
Y_insitu = chla_insitu_val
rmse = np.sqrt(np.mean((X_sat[1:] - Y_insitu[1:])**2))

fig = plt.figure(figsize=(10.5,7), dpi=600)
plt.xlabel('Satellite derived Chl-a [$mg/m^3$]')
plt.ylabel('In-situ Chl-a [$mg/m^3$]')
plt.title(f'Validation\nRMSE = {round(rmse, 2)}')
plt.scatter(X_sat[:], Y_insitu[:], color='k')
```

## Creating a time-series
Extracted information from satellite imagery can be converted into a Pandas dataframe, with which a time-series plotting can be done.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(14, 6))

# Add fitting curves.
ax.plot(chla_df_inlet['datetime'], chla_df_inlet['Chla'],
           'k-', alpha=0.8, label='Chla - RS')
inlet_dic = {
    'datetime': ['2019-05-14', '2019-06-24', '2019-07-18', '2019-08-20', '2019-09-12', '2019-10-11'],
    'Chla': [2.0985, 3.1390, 4.2744, 3.1478, 9.3652, 13.3576]
}
inlet_df = pd.DataFrame(inlet_dic)
inlet_df['datetime'] = pd.to_datetime(inlet_df['datetime'])

ax.scatter(inlet_df['datetime'], inlet_df['Chla'], c='None', edgecolors='k', s=200, label='Chla - 0m')
```


## Creating a mapping of a specific variable
We can effectively obtain accurate water area using the method proposed by Donchyts et al, 2016 and the JWP database.

```{code-cell} ipython3
gsw = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
occurrence = gsw.select('occurrence')

lake_kyoga_entire = ee.Geometry.Polygon([
        [
            [
                32.4, 1.7
            ],
            [
                32.4, 1.25
            ],
            [
                33.5, 1.25
            ],
            [
                33.5, 1.7
            ],
            [
                32.4, 1.7
            ]
        ]
    ])
feature = ee.Feature(lake_kyoga_entire, {})
roi = feature.geometry()

water_mask = occurrence.gt(50).clip(roi).selfMask()

jrc_fc = ras_to_vec(water_mask)
fc_largest = jrc_fc.limit(1, 'area', False)
```

```{code-cell} ipython3
Sen2ImageCollection = ee.ImageCollection('COPERNICUS/S2_SR')\
        .filterBounds(geom)\
        .filterDate(startDate, endDate)\
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)\
        .map(cloudMasking)

withWater = addWater(Sen2ImageCollection)
withWater = withWater.divide(10000)
updatedMaskedImage = withWater.select(S2_BANDS_2, S2_NAMES_2)
    
fig = plt.figure(figsize=(15, 10))
Chla = updatedMaskedImage.expression(equation,    {'RED2':updatedMaskedImage.select('red2'),'RED':updatedMaskedImage.select('red'), 'RED4':updatedMaskedImage.select('red4')}).rename('Chla')

mask = withWater.select('water').gt(0);
ChlaMask = Chla.updateMask(mask);
    
ChlaVisParam={
        'min':0,
        'max':chla_max,
        'palette':['152E13', '183815', '1A4216', '1C4D17', '1E5817', '206416', '237015', '257D14', '278A12', '29980F', '2BA60C', '2EB509', '31C405', '34D400', '71DD22', 'A5E544', 'D0EC66']
    };

region = [33.5, 1.25, 32.4, 1.7]
ax = cartoee.get_map(ChlaMask.clip(fc_largest.geometry()), region=region, vis_params=ChlaVisParam)
```


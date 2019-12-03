## About
The road classification program takes a road and building dataset, and classifies the roads into one of the four RIVM road types:
1. `Class 1`: a wide street canyon.
2. `Class 2`: a narrow street canyon.
3. `Class 3`: buildings on one side of the road.
4. `Class 4`: all other options.
The classification is based on ratio between distance and height from the medial axis of the road segment.

## Pre-requisites
This software runs on `Python3.x` and the following packages are necessary:
* `numpy`
* `shapely`
* `scipy`
* `Rtree`
* `pyhull`
* `fiona`
* `matplotlib`

## Usage
The program can be run by using `python classify_roads.py`.

### Parameters
Different parameters can be specified in the `params.json` file, which must be placed in the same folder as the `classify_roads.py` file. There are different options available:
* `roads`: provide a road dataset. Preferably downloaded from NWB (Nationaal Wegen Bestand).
* `buildings`: provide a buildings dataset. This data should contain buildings with an `id` and `height` value.
* `method`: the classification method to use. The options are: `average`, `weighted_average`, `raytracing`, and `raytracing_both`.
    * `average`: average the distance and heights of both sides of the roads, and use these values to classify.
    * `weighted_average`: use the length of the building facade to compute a weight factor for the distance and height of the building (based on Voronoi cells).
    * `raytracing`: use the side with the most buildings to create points along the road segment. Each point belongs to a building, and the points are classified independently. A voting system is used to classify the entire road segment.
    * `raytracing_both`: use both sides of the road and each point that gets created belongs to a building. A voting system is used to classify the entire road segment.
* `percentile`: provide the height percentile to use. Can be for example `roof-95` or `roof-99`. If no height percentiles are available, use the name of the `height` attribute field name.

### Suitable datasets
The datasets are preferably in .gpkg (GeoPackage format.) The road data from NWB is recommended, and for the buildings the BGT (Basisregistratie Grootschalige Topografie) is recommended. The BGT should be enriched with height data.

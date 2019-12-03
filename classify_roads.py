"""
Students:
Giulia Ceccarelli
Imke Lansky
Jinglan Li
Konstantinos Mastorakis
Wessel de Jongh

Summary:


Execution:


"""

import re
import json
import os.path
import sys
import random
from string import ascii_lowercase, ascii_uppercase
from math import sqrt, atan2, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.ops import nearest_points
from shapely.geometry import shape, Polygon, Point, LineString
from scipy.spatial import Voronoi, voronoi_plot_2d
from rtree import index
from pyhull.voronoi import VoronoiTess
import fiona
import time

# pip install rtree
# pip install osmnx
# install the following: https://github.com/mikedh/trimesh/issues/189


road_lengths = {'0-20': {}, '20-40': {}, '40-60': {}, '60-80': {},
                '80-100': {}, '100-': {}}

def distance(p_1, p_2):
    """
    Distance between two given points.
    """
    return sqrt((p_2[0] - p_1[0])**2 + (p_2[1] - p_1[1])**2)


def construct_bbox(all_points):
    """
    Construct the bounding box based on all points from the
    road and buildings that were discretised.
    """

    maximum = list(map(max, zip(*all_points)))
    minimum = list(map(min, zip(*all_points)))

    bbox = [(minimum[0], minimum[1]),
            (minimum[0], maximum[1]),
            (maximum[0], maximum[1]),
            (maximum[0], minimum[1]),
            (minimum[0], minimum[1])]

    return bbox


def get_buildings_in_buffer(buf, buildings, ids, idx):
    """
    Input the buffer polygon and building geometries to check
    if the building intersects with the buffer. Return all
    buildings within the buffer (based on ID). An R-tree
    is used to speed up things.
    """
    bld_in_buffer = {}

    for i in idx.intersection(buf.bounds):
        if buf.intersects(buildings[i]):
            bld_in_buffer[ids[i]] = buildings[i]

    return bld_in_buffer


def get_equidistant_points(p_1, p_2, parts):
    """
    Source:
    https://stackoverflow.com/questions/47443037/equidistant-points-between-two-points-in-python
    """
    return list(zip(np.linspace(p_1[0], p_2[0], parts + 1),
                    np.linspace(p_1[1], p_2[1], parts + 1)))


def discretise_road(road_coords):
    """
    Discretise all parts of the road segment. Take two consecutive coordinates
    in the list and go over all the points.
    """
    eq_pts_road = []

    for j in range(len(road_coords) - 1):
        dist = distance(road_coords[j], road_coords[j + 1])
        eq_pts_road.extend(get_equidistant_points(road_coords[j], road_coords[j + 1],
                                                  int(dist / 4)))
    return eq_pts_road


def discretise_bbox(pol):
    """
    Discretise the extend of the bounding box which is derived from the
    maximum and minimum coordinates of all geometries.
    """

    eq_points_pol = []

    for i in range(len(pol) - 1):
        first = pol[i]
        second = pol[i + 1]
        dist = distance(first, second)

        eq_pts = get_equidistant_points(first, second, int(dist / 2))
        eq_points_pol.extend(eq_pts)

    return eq_points_pol


def discretise_buildings(buildings):
    """
    Discretise and entire input list of buildings. Output a flat list with
    all coordinates to input in the voronoi function. Also create a dictionary
    that contains all coordinates per building ID.
    """

    eq_pts_buildings = {}
    eq_points_flat = []

    for b_id, building in buildings.items():
        eq_pts_buildings[b_id] = []

        # Building is of type MultiPolygon.
        if building.geom_type == "MultiPolygon":

            # Go over every sub-polygon in the MultiPolygon.
            for sub_polygon in building:
                sub_pol_pts = list(sub_polygon.exterior.coords)

                # Discretise for every sub-polygon its boundary into points.
                for j in range(len(sub_pol_pts) - 1):
                    first = sub_pol_pts[j]
                    second = sub_pol_pts[j + 1]
                    dist = distance(first, second)

                    eq_pts = get_equidistant_points(first, second, int(dist / 2))
                    eq_pts_buildings[b_id].extend(eq_pts)
                    eq_points_flat.extend(eq_pts)

        # Building is of type Polygon.
        elif building.geom_type == 'Polygon':
            bld_pts = list(building.exterior.coords)

            # Discretise polygon boundary into points.
            for j in range(len(bld_pts) - 1):
                first = bld_pts[j]
                second = bld_pts[j + 1]
                dist = distance(first, second)

                eq_pts = get_equidistant_points(first, second, int(dist / 2))
                eq_pts_buildings[b_id].extend(eq_pts)
                eq_points_flat.extend(eq_pts)

        # Something else happened.
        else:
            print("Polygon not of type Polygon or MultiPolygon - Discretising")

    return eq_pts_buildings, eq_points_flat


def get_data(file, percentile, f_type):
    """
    Extract data from the input files.
    Return the ids and geometries of the features in the file.
    Create an index on the buildings (R-tree).
    """

    # Check if the file exists.
    if not os.path.isfile(file):
        print("File:", file, "does not exist!")
        sys.exit()

    data = {}
    ids = []
    geoms = []
    idx = index.Index()
    count = 0

    with fiona.open(file) as filepointer:
        for feature in filepointer:
            data[feature['id']] = {'geom': shape(feature['geometry'])}

            # For buildings store some extra information.
            if f_type == "buildings":
                idx.insert(count, list(shape(feature['geometry']).bounds))
                ids.append(feature['id'])
                geoms.append(shape(feature['geometry']))
                data[feature['id']]['height'] = feature['properties'][percentile]
                count += 1

    if f_type == "roads":
        return data

    return data, geoms, ids, idx


def visualise_buildings(buildings, colour):
    """
    Visualise buildings of given list. Check if the building is a MultiPolygon
    or Polygon - extract necessary information accordingly.
    """

    for bld in buildings.values():
        # colour = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

        if bld.geom_type == "MultiPolygon":
            sub_polygons = [sub.exterior.xy for sub in bld]

            for x_coord, y_coord in sub_polygons:
                plt.plot(x_coord, y_coord, c=colour, linewidth=1)

        elif bld.geom_type == "Polygon":
            b_x, b_y = bld.exterior.xy
            plt.plot(b_x, b_y, c=colour, linewidth=1)

        else:
            print("Cannot visualise buildings - No MultiPolygon or Polygon type")


def visualise_buffer(buf):
    """
    Plot the buffer geometry.
    """
    b_x, b_y = buf.exterior.xy
    plt.plot(b_x, b_y, c='royalblue', linestyle='--', linewidth=1)


def visualise_road(road):
    """
    Plot the road geometry.
    """
    r_x, r_y = road.xy
    plt.plot(r_x, r_y, c='maroon', linewidth=1)


def visualise_points(pts):
    """
    Plot the equidistant points.
    """
    plt.scatter(*zip(*pts), c='grey', s=4)


def find_road_building_relations(voronoi, pts_line, pts_left, pts_right, road_crds):
    """
    Find all road IDs facing the current road segment. Makes use of the indices
    of the points between which a voronoi ridge lies. This can be used to check
    if a road and building are facing each other.
    """

    r_first = road_crds[0]
    r_last = road_crds[-1]

    ridge_points = voronoi.ridges

    b_ids_left = []
    b_ids_right = []

    # Store in a dictionary for each road segment ID, which building IDs
    # belong to the road.
    for pair in ridge_points:
        p_1 = voronoi.points[pair[0]]
        p_2 = voronoi.points[pair[1]]

        # Exlude end points of road for voronoi - limit wrong classifications
        # of buildings facing a road at the end of the street.
        if p_1 == r_first or p_1 == r_last or p_2 == r_first or p_2 == r_last:
            continue

        # Point one belongs to the road segment.
        if p_1 in pts_line:

            # Point two belongs to a building.
            fids_left = [fid for fid, bld_coords in pts_left.items() if p_2 in bld_coords]
            fids_right = [fid for fid, bld_coords in pts_right.items() if p_2 in bld_coords]
            if fids_left:
                b_ids_left.extend(fids_left)
            elif fids_right:
                b_ids_right.extend(fids_right)

        # Point two belongs to a road segment.
        elif p_2 in pts_line:

            # Point one belongs to a building.
            fids_left = [fid for fid, bld_coords in pts_left.items() if p_1 in bld_coords]
            fids_right = [fid for fid, bld_coords in pts_right.items() if p_1 in bld_coords]
            if fids_left:
                b_ids_left.extend(fids_left)
            elif fids_right:
                b_ids_right.extend(fids_right)

    # Count the number of cells that touch the road for each building,
    # which can then be used as a weight.
    left_weights = {n_left_bld:b_ids_left.count(n_left_bld) for n_left_bld in b_ids_left}
    right_weights = {n_right_bld:b_ids_right.count(n_right_bld) for n_right_bld in b_ids_right}

    # Remove the duplicate buildings corresponding to a road segment and return the result.
    return list(set(b_ids_left)), list(set(b_ids_right)), left_weights, right_weights


def avg_values(buildings_buf, building_data, road):
    """
    Get some really simple statistics on a set of buildings
    given a road segment. Return average building height and
    average distance from the buildings to the road.
    """

    if not buildings_buf:
        return 0, 0

    avg_dist = 0
    avg_height = 0

    for bid in buildings_buf:
        avg_dist += building_data[bid]['geom'].distance(road)

        # Sometimes no data available in the dataset.
        if building_data[bid]['height'] is not None:
            avg_height += building_data[bid]['height']

    avg_dist /= len(buildings_buf)
    avg_height /= len(buildings_buf)

    return avg_dist, avg_height


def weighted_avg_values(buildings_buf, building_data, weight, road):
    """
    Get some really simple statistics on a set of buildings
    given a road segment. Return average building height and
    average distance from the buildings to the road.
    """

    total_weight = (sum(weight.values()))

    if not buildings_buf:
        return 0, 0

    weighted_avg_dist = 0
    weighted_avg_height = 0

    for bid in buildings_buf:
        bld_distance = building_data[bid]['geom'].distance(road)
        weighted_avg_dist += (bld_distance * (weight[bid] / total_weight))

        # Sometimes there is no data available in the dataset,
        # we skip these values.
        if building_data[bid]['height'] is not None:
            bld_height = building_data[bid]['height']
            weighted_avg_height += (bld_height * (weight[bid] / total_weight))

    weighted_avg_dist /= len(buildings_buf)
    weighted_avg_height /= len(buildings_buf)

    return weighted_avg_dist, weighted_avg_height


def write_roads(road_data, filename):
    """
    Write the road data dictionary to a CSV file which can be visualised
    in a GIS program.
    """

    with open(filename, 'w') as filepointer:
        filepointer.write("fid; geometry; class\n")

        for key in road_data.keys():
            filepointer.write("%s;%s;%s\n"%(key, road_data[key]['geom'].wkt,
                                            road_data[key]['class']))

    print("Written road data output to file:", filename)


def write_points(points_data, filename):
    """
    Custom function to export the classified points to CSV for visualization purposes.
    """

    with open(filename, 'w') as filepointer:
        filepointer.write("fid; geometry; class\n")
        for key in points_data.keys():
            for i in points_data[key]['geom'].keys():
                filepointer.write("%s;%s;%s\n"%(key, points_data[key]['geom'][i].wkt,
                                                points_data[key]['class'][i]))

    print("Written point data output to file:", filename)


def check_road_length(length):
    """
    Determine the key for where to store the road length.
    """

    key = ''

    if length <= 20:
        key = '0-20'
    elif 20 < length <= 40:
        key = '20-40'
    elif 40 < length <= 60:
        key = '40-60'
    elif 60 < length <= 80:
        key = '60-80'
    elif 80 < length <= 100:
        key = '80-100'
    else:
        key = '100-'

    return key


def get_buffer_parts(r_id, road_simple):
    """
    For a road compute the buffer(s). If the road consists of two
    coordinates, simply create a buffer on both sides by using an offset.
    If there are more than two coordiantes, the road is split into parts
    and each of these parts gets a separate buffer. A dictionary with all
    buffers is returned.
    """

    # Keep a global list of all the road lengths. Includes lengths of the
    # sub roads that are created.
    global road_lengths

    road_simple_coords = list(road_simple.coords)
    buffers = {}
    extensions = list(ascii_lowercase) + list(ascii_uppercase)

    # The road consists of only two points. Directly offset it and create the two buffers.
    if len(road_simple_coords) == 2:
        left_buf = road_simple.parallel_offset(60, side='left', resolution=16, join_style=2)
        right_buf = road_simple.parallel_offset(60, side='right', resolution=16, join_style=2)

        buffers[r_id] = {}
        buffers[r_id]['left'] = Polygon([road_simple.coords[0], road_simple.coords[1],
                                         left_buf.coords[1], left_buf.coords[0]])
        buffers[r_id]['right'] = Polygon([road_simple.coords[0], road_simple.coords[1],
                                          right_buf.coords[0], right_buf.coords[1]])
        buffers[r_id]['new_road'] = road_simple

        key = check_road_length(road_simple.length)
        road_lengths[key][r_id] = road_simple.length

    # The road is more complex. Create buffers for each sub-segment of the road.
    elif len(road_simple_coords) > 2:

        # Iterate over every sub-segment, create a linestring out of it and offset it.
        # Every sub-segment gets the road_id +  a letter out of the extensions list to
        # create a unique id.
        for i in range(len(road_simple_coords) - 1):
            new_id = str(r_id) + extensions[i]
            buffers[new_id] = {}
            sub_road = LineString([Point(road_simple_coords[i]),
                                   Point(road_simple_coords[i + 1])])

            left_buf = sub_road.parallel_offset(60, side='left', resolution=16, join_style=2)
            right_buf = sub_road.parallel_offset(60, side='right', resolution=16, join_style=2)

            buffers[new_id] = {}
            buffers[new_id]['left'] = Polygon([sub_road.coords[0], sub_road.coords[1],
                                               left_buf.coords[1], left_buf.coords[0]])
            buffers[new_id]['right'] = Polygon([sub_road.coords[0], sub_road.coords[1],
                                                right_buf.coords[0], right_buf.coords[1]])
            buffers[new_id]['new_road'] = sub_road

            key = check_road_length(sub_road.length)
            road_lengths[key][new_id] = sub_road.length

    return buffers


def classify_segment(height_r, dist_r, height_l, dist_l):
    """
    Classify a road segment based on height and distance values.
    This function is used by the averaging method.
    """

    # Sometimes the buildings in the dataset don't have any data.
    # Function returns 0 then and we just classify as 4 because we
    # cannot really say what to classify it otherwise.
    if (height_r == 0) and (height_l == 0):
        r_class = 4

    # Class 3
    elif (height_l == 0) and (height_r != 0):
        if (dist_r / height_r) < 3:
            r_class = 3
        else:
            r_class = 4

    # Class 3
    elif (height_r == 0) and (height_l != 0):
        if (dist_l / height_l) < 3:
            r_class = 3
        else:
            r_class = 4

    # Class 1
    elif ((dist_r / height_r) < 3) and (1.5 < (dist_l / height_l) < 3):
        r_class = 1

    # Class 1
    elif ((dist_l / height_l) < 3) and \
      (1.5 < (dist_r / height_r) < 3):
        r_class = 1

    # Class 2
    elif ((dist_l / height_l) < 1.5) and ((dist_r / height_r) < 1.5):
        r_class = 2

    # Class 4
    else:
        r_class = 4

    return r_class


def get_points_raytracing(count, side, other_side, building_data, road_part, points_classified):
    """
    Get all the point created by the raytracing for the given road segment.
    Returns a dictionary with the distances and heights on the left and right side.
    """
    bld_dist_heights = {}
    i = count

    for ids in side:

        # Find the closest points between every building and its segments
        response = nearest_points(building_data[ids]['geom'], road_part)
        bld_dist_heights[ids] = []

        # Store point coordinates in dictionary
        points_classified['geom'][i] = response[1]

        # Calculate distance and height of corresponding building
        dist_l = building_data[ids]['geom'].distance(road_part)
        height_l = building_data[ids]['height']

        bld_dist_heights[ids].append(dist_l)
        bld_dist_heights[ids].append(height_l)

        # Calculate extended point to a distance of 61 meters from the road
        # axis and create line segment
        angle = atan2((response[1].x - response[0].x), (response[1].y - response[0].y))
        extended_point = (response[1].x + 61 * sin(angle), response[1].y + 61 * cos(angle))
        extended_line_segment = LineString([(response[1].x, response[1].y), extended_point])

        # Try finding intersections on the other side of the road
        # with the extended line segments. Store the distances and heights.
        if other_side:
            for ids_other in other_side:
                if extended_line_segment.intersects(building_data[ids_other]['geom']):
                    dist_r = building_data[ids_other]['geom'].distance(road_part)
                    height_r = building_data[ids_other]['height']
                    break
                else:
                    dist_r = 0
                    height_r = 0

            bld_dist_heights[ids].append(dist_r)
            bld_dist_heights[ids].append(height_r)

        i += 1

    return bld_dist_heights, points_classified


def classify_points_raytracing(bld_dist_heights, points_classified):
    """
    For each point created on the line segment by the raytracing method,
    classify them based on the given distance and heights of the buildings
    beloning to that point.
    """

    for i, building in enumerate(bld_dist_heights):
        dist_l = bld_dist_heights[building][0]
        height_l = bld_dist_heights[building][1]

        # Buildings on both sides.
        if len(bld_dist_heights[building]) == 4:
            dist_r = bld_dist_heights[building][2]
            height_r = bld_dist_heights[building][3]

            # Checking if height in building is None, if it is assign "None" as class
            if ((height_r) is None) or ((height_l) is None):
                points_classified['class'][i] = "None"

            else:
                # No buildings either side, class 4.
                if (height_r == 0) and (height_l == 0):
                    points_classified['class'][i] = 4
                    continue

                # Class 3
                if height_l == 0 and height_r != 0:
                    if (dist_r / height_r) < 3:
                        points_classified['class'][i] = 3
                    else:
                        points_classified['class'][i] = 4

                # Class 3
                elif height_r == 0 and height_l != 0:
                    if (dist_l / height_l) < 3:
                        points_classified['class'][i] = 3
                    else:
                        points_classified['class'][i] = 4

                # Class 1
                elif ((dist_r / height_r) < 3) and (1.5 < (dist_l / height_l) < 3):
                    points_classified['class'][i] = 1

                # Class 1
                elif ((dist_l / height_l) < 3) and (1.5 < (dist_r / height_r) < 3):
                    points_classified['class'][i] = 1

                # Class 2
                elif ((dist_l / height_l) < 1.5) and ((dist_r / height_r) < 1.5):
                    points_classified['class'][i] = 2

                # Class 4
                else:
                    points_classified['class'][i] = 4

        # Only buildings on one side.
        else:
            if (height_l is None) or (height_l == 0):
                points_classified['class'][i] = "None"

            else:
                if (dist_l / height_l) < 3:
                    points_classified['class'][i] = 3
                else:
                    points_classified['class'][i] = 4

    return points_classified


def classify_segment_raytracing(points_classified):
    """
    After all points in a segment are classified, the
    street is classified according to the class that
    occurs most often.
    """

    ones, twos, threes, fours = 0, 0, 0, 0
    for j in points_classified['class']:
        if points_classified['class'][j] == 1:
            ones += 1
        elif points_classified['class'][j] == 2:
            twos += 1
        elif points_classified['class'][j] == 3:
            threes += 1
        elif points_classified['class'][j] == 4:
            fours += 1

    categories = [ones, twos, threes, fours]

    # Finding the most classified category
    str_type = categories.index(max(categories)) + 1

    # In case there are no buildings in any side automatically classify as category 4
    # else according to the most classified type
    if max(categories) == 0:
        r_class = 4
    else:
        r_class = str_type

    return r_class


def classify(method, building_data, blds_left, blds_right, weight_left, weight_right,
             road, points_classified):
    """
    Classify the road segment based on the chose classification method.
    """

    r_class = None

    # Check which kind of classification to use. Call the appropriate
    # function accordingly.
    if method == 'average':
        avg_dist_left, avg_height_left = avg_values(blds_left, building_data, road)
        avg_dist_right, avg_height_right = avg_values(blds_right, building_data, road)
        r_class = classify_segment(avg_height_right, avg_dist_right, avg_height_left, avg_dist_left)
        return r_class

    if method == 'weighted_average':
        avg_dist_left, avg_height_left = weighted_avg_values(blds_left, building_data,
                                                             weight_left, road)
        avg_dist_right, avg_height_right = weighted_avg_values(blds_right, building_data,
                                                               weight_right, road)
        r_class = classify_segment(avg_height_right, avg_dist_right, avg_height_left, avg_dist_left)
        return r_class

    if method == 'raytracing':
        side = blds_right
        other_side = blds_left

        if len(blds_left) > len(blds_right):
            side = blds_left
            other_side = blds_right

        buildings_dist_heights, points_classified = get_points_raytracing(0, side, other_side,
                                                                          building_data, road,
                                                                          points_classified)
        p_class = classify_points_raytracing(buildings_dist_heights, points_classified)
        r_class = classify_segment_raytracing(p_class)

        return r_class, p_class

    if method == 'raytracing_both':

        buildings_dist_heights_left, points_classified = \
          get_points_raytracing(0, blds_left, blds_right, building_data, road, points_classified)

        buildings_dist_heights_right, points_classified = \
          get_points_raytracing(len(points_classified['geom']), blds_right, blds_left,
                                building_data, road, points_classified)

        #possible problem: dictionary may not be sorted... maybe points don't matc
        buildings_dist_heights = {**buildings_dist_heights_left, **buildings_dist_heights_right}

        p_class = classify_points_raytracing(buildings_dist_heights, points_classified)
        r_class = classify_segment_raytracing(p_class)

        return r_class, p_class

    print("This classification method doest not exist!")
    sys.exit()


def check_road_segment(r_id, road, building_data, bld_geoms, bld_ids, method, idx):
    """
    This function runs for every road segment the main logic.
    It creates a buffer, finds the buildings in the buffer,
    constructs abounding box, creates the voronoi, finds relationshipds
    between road segments and buildings, and the calls the classification.
    """

    road_to_building = {}
    classification = {}
    points_classified = {}

    # Create equidistant points along the road segment.
    road_coords = list(road.coords)
    eq_pts_road = discretise_road(road_coords)

    # Create buffer around the road (60m) and return all of its coordinates.
    # Discretise the buffer based on these coordinates for bounding the Voronoi.
    road_simple = road.simplify(5, preserve_topology=True)

    # Get the buffers for the roads and road sub-segments in the case the
    # road is too complex.
    sub_buffers = get_buffer_parts(r_id, road_simple)

    for new_id in sub_buffers:

        points_classified[new_id] = {}
        points_classified[new_id]['geom'] = {}
        points_classified[new_id]['class'] = {}

        road_part = sub_buffers[new_id]['new_road']
        classification[new_id] = {'geom': road_part}
        road_to_building[new_id] = {}

        # Find the buildings in the 60m buffer.
        blds_left_buffer = get_buildings_in_buffer(sub_buffers[new_id]['left'],
                                                   bld_geoms, bld_ids, idx)
        blds_right_buffer = get_buildings_in_buffer(sub_buffers[new_id]['right'],
                                                    bld_geoms, bld_ids, idx)

        # No buildings on either side of the road, go to the next road segment.
        if not blds_left_buffer and not blds_right_buffer:
            classification[new_id]['class'] = 4
            continue

        # Discretise the buildings for the voronoi.
        eq_pts_blds_left, eq_pts_blds_left_flat = discretise_buildings(blds_left_buffer)
        eq_pts_blds_right, eq_pts_blds_right_flat = discretise_buildings(blds_right_buffer)

        # Create voronoi diagram out of the building, road and bounding box points.
        bbox = construct_bbox(eq_pts_road + eq_pts_blds_left_flat + eq_pts_blds_right_flat)
        eq_pts_bbox = discretise_bbox(bbox)

        # Create the list with points for the voronoi diagram.
        vor_pts = eq_pts_road + eq_pts_blds_left_flat + eq_pts_blds_right_flat + eq_pts_bbox

        # Sometimes the library errors without a clear reason. In these cases
        # we set the class of the road segment to None.
        try:
            vor = VoronoiTess(vor_pts, add_bounding_box=False)
        except:
            print("Could not create Voronoi for this segment, id:", new_id)
            classification[new_id]['class'] = 'None'
            continue

        # Find the buildings facing the road for the left and right side.
        blds_left, blds_right, weights_left, weights_right = \
          find_road_building_relations(vor, eq_pts_road, eq_pts_blds_left,
                                       eq_pts_blds_right, road_coords)

        road_to_building[new_id]['left'] = blds_left
        road_to_building[new_id]['right'] = blds_right

        # general: 21283, 27058, 27267, 28822
        # voronoi column plots: 4269, 4270, 5811, 6283, 22841
        if new_id in ('19662'):
            vor = Voronoi(vor_pts)
            sub_left = {k: blds_left_buffer[k] for k in road_to_building[new_id]['left']}
            sub_right = {k: blds_right_buffer[k] for k in road_to_building[new_id]['right']}
            # visualise_buildings(blds_left_buffer, 'darkturquoise')
            # visualise_buildings(blds_right_buffer, 'seagreen')
            # visualise_buildings(sub_left, 'green')
            # visualise_buildings(sub_right, 'green')
            # visualise_buffer(sub_buffers[new_id]['left'])
            # visualise_buffer(sub_buffers[new_id]['right'])
            # visualise_road(road_part)
            # plt.title("Buildings within 60m of the road", fontweight="bold", fontsize=10)

            # legend_elements = [Line2D([0], [0], color='maroon', lw=1, label='Road'),
            #                    Line2D([0], [0], color='royalblue', lw=1, label='Buffer', linestyle='--'),
            #                    Line2D([0], [0], color='darkturquoise', lw=1, label='Buildings left'),
            #                    Line2D([0], [0], color='seagreen', lw=1, label='Buildings right')]

            # plt.legend(handles=legend_elements, loc='upper left', fontsize='small')
            # plt.xlabel("x-coordinate*")
            # plt.ylabel("y-coordinate*")
            # plt.tight_layout()
            # plt.annotate('* In RD NEW', xy=(1, 0), xycoords='figure fraction',
            #              xytext=(-15, 5), textcoords='offset pixels',
            #              horizontalalignment='right',
            #              verticalalignment='bottom', fontsize=7)
            # plt.savefig(new_id + "_layout.pdf", dpi=300, format='pdf')

            # fig = plt.figure()
            # fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='lightgrey', line_width=1,
            #                 line_alpha=1, point_size=4)

            # ax = fig.add_subplot(1, 1, 1)
            # visualise_buildings(blds_left_buffer, 'red')
            # visualise_buildings(blds_right_buffer, 'red')
            # visualise_buildings(sub_left, 'green')
            # visualise_buildings(sub_right, 'green')
            # visualise_road(road_part)
            # ax.set_title("Voronoi plot showing building cells touching road cells",
            #           fontweight="bold", fontsize=10)
            # ax.set_xlabel("x-coordinate*")
            # ax.set_ylabel("y-coordinate*")
            # ax.set_xlim(122235, 122287)
            # ax.set_ylim(486325, 486381)
            # ax.get_yaxis().get_major_formatter().set_useOffset(False)
            # ax.get_xaxis().get_major_formatter().set_useOffset(False)
            # ax.ticklabel_format(axis='both', style='plain')
            # legend_elements = [Line2D([0], [0], color='maroon', lw=1, label='Road'),
            #                    Line2D([0], [0], color='green', lw=1, label='Facing road'),
            #                    Line2D([0], [0], color='red', lw=1, label='Not facing road')]
            # ax.legend(handles=legend_elements, loc='upper left', fontsize='small')
            # fig.tight_layout()
            # ax.annotate('* In RD NEW', xy=(1, 0), xycoords='figure fraction',
            #              xytext=(-15, 5), textcoords='offset pixels',
            #              horizontalalignment='right',
            #              verticalalignment='bottom', fontsize=7)
            # # plt.show()
            # plt.savefig(new_id + "_voronoi_bld_cells.pdf", dpi=300, format='pdf')

        # Get the classification for the given road segment and the points
        # beloning to that road segment if the ray-tracing option is used.
        if method in ('raytracing', 'raytracing_both'):
            classification[new_id]['class'], points_classified[new_id] = \
              classify(method, building_data, blds_left, blds_right, weights_left,
                       weights_right, road_part, points_classified[new_id])

        else:
            classification[new_id]['class'] = classify(method, building_data, blds_left,
                                                       blds_right, weights_left, weights_right,
                                                       road_part, points_classified[new_id])

        # Extract a temporary sub dictionary for buildings that are facing the road
        # and visualise them.
        # sub_left = {k: blds_left_buffer[k] for k in road_to_building[new_id]['left']}
        # sub_right = {k: blds_right_buffer[k] for k in road_to_building[new_id]['right']}
        # voronoi_plot_2d(vor, show_vertices=False, line_colors='peru', line_width=1,
        #                   line_alpha=0.6, point_size=1)
        # visualise_buildings(sub_left)
        # visualise_buildings(sub_right)
        # visualise_road(road)
        # plt.show()

    return classification, points_classified


def main():
    starttime = time.time()
    """
    Main logic of the program. Call all functions and get user inputs.
    """

    # Read parameters from file 'params.json' (same folder as this file).
    try:
        jparams = json.load(open('params.json'))
    except FileNotFoundError:
        print("ERROR: Something is wrong with the params.json file.")
        sys.exit()

    f_roads = jparams['roads']
    f_buildings = jparams['buildings']
    method = jparams['method']
    percentile = jparams['percentile']

    road_data = get_data(f_roads, percentile, "roads")
    building_data, bld_geoms, bld_ids, idx = get_data(f_buildings, percentile, "buildings")

    classification, points_classified = {}, {}

    # For every road segment, repeat the whole process of the classification.
    for r_id, r_geom in road_data.items():
        road = r_geom['geom']

        # Sometimes roads can be of type MultiLineString, then we need to loop
        # over all the individual parts making up the road.
        if road.geom_type == "LineString":
            r_class, p_class = check_road_segment(r_id, road, building_data,
                                                  bld_geoms, bld_ids, method, idx)
            classification.update(r_class)
            points_classified.update(p_class)

        elif road.geom_type == "MultiLineString":
            for part in road:
                r_class, p_class = check_road_segment(r_id, part, building_data,
                                                      bld_geoms, bld_ids, method, idx)
                classification.update(r_class)
                points_classified.update(p_class)

        else:
            print("Some strange geometry type for the road!")
            sys.exit()

    # Extract filename from path, remove extension and use it in the
    # file that is saved to disk.
    split_f_road = re.findall(r"\w+(?:\.\w+)*$", f_roads)[0].rsplit('.', 1)
    filename_roads = method + "_" + percentile + "_" + split_f_road[0] + ".csv"

    write_roads(classification, filename_roads)

    if method in ('raytracing', 'raytracing_both'):
        filename_points = method + "_" + percentile + "_" + split_f_road[0] + "_Points.csv"
        write_points(points_classified, filename_points)

    # Print a statistic on the different lengths of roads in the given
    # dataset.
    endtime = time.time()
    duration = endtime-starttime
    print ("Runtime: ", round(duration, 2), "s")

    for key, value in road_lengths.items():
        print(key, ":", len(value))


if __name__ == '__main__':
    main()

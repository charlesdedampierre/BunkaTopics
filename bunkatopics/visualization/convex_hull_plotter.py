import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull


def get_density_hull_coords(points, outlier_percentile=75, interpolate_curve=True):
    """
    Calculate hull after removing outliers based on distance from cluster center.
    """
    # Find centroid of points
    centroid = np.mean(points, axis=0)

    # Calculate distance of each point from centroid
    distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))

    # Determine threshold for outliers
    threshold = np.percentile(distances, outlier_percentile)

    # Filter out outliers
    inliers = points[distances <= threshold]

    # Calculate convex hull on inliers only
    hull = ConvexHull(inliers)

    # Get hull coordinates and interpolate as before
    x_hull = np.append(inliers[hull.vertices, 0], inliers[hull.vertices, 0][0])
    y_hull = np.append(inliers[hull.vertices, 1], inliers[hull.vertices, 1][0])

    if interpolate_curve:
        # Same interpolation code as your original function
        dist = np.sqrt(
            (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
        )
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
    else:
        interp_x = x_hull
        interp_y = y_hull

    return interp_x, interp_y


def get_convex_hull_coord(points: np.array, interpolate_curve: bool = True) -> tuple:
    """
    Calculate the coordinates of the convex hull for a set of points.

    Args:
        points (np.array): Array of points, where each row is [x, y].
        interpolate_curve (bool): Whether to interpolate the convex hull.

    Returns:
        tuple: Tuple containing interpolated x and y coordinates of the convex hull.
    """
    # Calculate the convex hull of the points
    hull = ConvexHull(points)

    # Get the x and y coordinates of the convex hull vertices
    x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
    y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])

    if interpolate_curve:
        # Calculate distances between consecutive points on the convex hull
        dist = np.sqrt(
            (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
        )

        # Calculate the cumulative distance along the convex hull
        dist_along = np.concatenate(([0], dist.cumsum()))

        # Use spline interpolation to generate interpolated points
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
    else:
        # If interpolation is not needed, use the original convex hull points
        interp_x = x_hull
        interp_y = y_hull

    return interp_x, interp_y

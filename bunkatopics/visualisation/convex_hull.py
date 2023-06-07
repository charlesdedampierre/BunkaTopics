import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull


def get_convex_hull_coord(points: np.array, interpolate_curve: bool = True):
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
    y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])

    if interpolate_curve:
        # interpolate
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

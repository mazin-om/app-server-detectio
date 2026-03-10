"""
Coordinate calculation utilities for estimating GPS coordinates of detected objects.
Uses bearing-based offset from the device's GPS position.
"""

import math
import logging

logger = logging.getLogger(__name__)


def offset_gps(lat, lon, offset_north_m, offset_east_m):
    """
    Calculate GPS coordinates from offset in meters.

    Args:
        lat, lon: Starting GPS coordinates (degrees)
        offset_north_m: North offset in meters (positive = north)
        offset_east_m: East offset in meters (positive = east)

    Returns:
        (new_lat, new_lon) in degrees
    """
    R = 6371000  # Earth radius in meters

    lat_rad = math.radians(lat)

    if abs(lat) >= 90.0:
        raise ValueError(f"Invalid latitude: {lat}")

    dlat = offset_north_m / R

    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-10:
        raise ValueError(f"Latitude too close to pole: {lat}")

    dlon = offset_east_m / (R * cos_lat)

    new_lat = lat + math.degrees(dlat)
    new_lon = lon + math.degrees(dlon)

    if not (-90 <= new_lat <= 90) or not (-180 <= new_lon <= 180):
        raise ValueError(f"Calculated coordinates out of range: ({new_lat}, {new_lon})")

    return new_lat, new_lon

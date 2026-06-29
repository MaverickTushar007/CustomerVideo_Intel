from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2

class ZoneMapper:
    """
    Handles floorplan zone mapping, spatial containment, and homography matrix calculations.
    """
    def __init__(self, zones: Dict[str, List[Tuple[float, float]]], homography_matrix: Optional[List[List[float]]] = None):
        """
        zones: dict of zone_id -> list of polygon vertices (x, y)
        homography_matrix: 3x3 homography matrix list for mapping camera view to top-down coordinates
        """
        self.zones = zones
        self.h_matrix = np.array(homography_matrix) if homography_matrix is not None else None

    def map_pixel_to_floor(self, pixel_coord: Tuple[float, float]) -> Tuple[float, float]:
        """
        Maps a 2D camera pixel coordinate to 2D top-down floor coordinates using the homography matrix.
        """
        if self.h_matrix is None:
            # Return identity mapping if homography matrix is not configured
            return pixel_coord
        
        src_pt = np.array([[[pixel_coord[0], pixel_coord[1]]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, self.h_matrix)
        floor_x = float(dst_pt[0][0][0])
        floor_y = float(dst_pt[0][0][1])
        return (floor_x, floor_y)

    def is_inside_zone(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Ray-casting algorithm to determine if a point is inside a polygon.
        """
        x, y = point
        inside = False
        n = len(polygon)
        if n < 3:
            return False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_zone_for_bbox(self, bbox: List[float]) -> Optional[str]:
        """
        Determines the zone for a bounding box. Uses the bottom-center point of the bounding box.
        """
        x1, y1, x2, y2 = bbox
        bottom_center = ((x1 + x2) / 2.0, y2)
        
        # Optionally project to floor space first if homography is enabled
        target_point = self.map_pixel_to_floor(bottom_center)

        for zone_id, polygon in self.zones.items():
            if self.is_inside_zone(target_point, polygon):
                return zone_id
        return None

import numpy as np

class GrabbableObject:
    def __init__(self, position=None, rotation=None, scale=None):
        # 3D position vector
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        # Rotation in degrees around x, y, z axes
        self.rotation = np.array(rotation if rotation is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        # Scale along x, y, z
        self.scale = np.array(scale if scale is not None else [1.0, 1.0, 1.0], dtype=np.float32)
        # Flag for whether this object is selected
        self.selected = False

    # --------------------------
    # Transformations
    # --------------------------
    def translate(self, delta):
        """Move object by delta [dx, dy, dz]"""
        self.position += np.array(delta, dtype=np.float32)

    def rotate(self, delta):
        """Rotate object by delta [dx, dy, dz] in degrees"""
        self.rotation += np.array(delta, dtype=np.float32)

    def scale_object(self, factor):
        """Scale object by factor [sx, sy, sz]"""
        self.scale *= np.array(factor, dtype=np.float32)

    # --------------------------
    # Placeholder render
    # --------------------------
    def render(self):
        """Override this method in derived classes to draw the object"""
        raise NotImplementedError("Render method must be implemented in subclass")

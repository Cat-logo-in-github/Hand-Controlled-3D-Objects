import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class Renderer:
    """
    Handles OpenGL context, scene rendering, and object drawing
    """
    def __init__(self, width=640, height=480, title="AR Powered Designer"):
        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        self.width = width
        self.height = height
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Camera / Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # Scene objects
        self.objects = []

    # --------------------------
    # Object management
    # --------------------------
    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)

    # --------------------------
    # Rendering
    # --------------------------
    def render_scene(self, camera_transform=np.eye(4)):
        """
        Render all objects in the scene
        :param camera_transform: 4x4 numpy array for camera (optional)
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Apply camera transform
        glTranslatef(0, 0, -5)  # simple fixed camera
        glMultMatrixf(camera_transform.T)  # OpenGL expects column-major

        # Draw objects
        for obj in self.objects:
            obj.render()  # <-- changed from draw() to render()

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # --------------------------
    # Coordinate mapping
    # --------------------------
    def normalized_to_world(self, x_norm, y_norm, z_norm, z_range=(-2, 2)):
        """
        Convert normalized [0-1] coordinates to OpenGL world coordinates
        :param x_norm, y_norm, z_norm: normalized
        :param z_range: min/max Z for mapping depth
        :return: np.array([x, y, z])
        """
        x = (x_norm - 0.5) * 4.0  # map 0-1 to -2 to 2
        y = (y_norm - 0.5) * 3.0  # map 0-1 to -1.5 to 1.5
        z = z_range[0] + z_norm * (z_range[1] - z_range[0])
        return np.array([x, y, z], dtype=np.float32)

    # --------------------------
    # Window utilities
    # --------------------------
    def should_close(self):
        return glfw.window_should_close(self.window)

    def terminate(self):
        glfw.terminate()

from OpenGL.GL import *
from .grabbable_object import GrabbableObject

class Cube(GrabbableObject):
    def __init__(self, size=1.0, **kwargs):
        super().__init__(**kwargs)
        self.size = size  # cube size

    def render(self):
        """Render cube using OpenGL"""
        glPushMatrix()
        # Apply transformations
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)
        glScalef(self.size, self.size, self.size)

        # Draw cube
        glBegin(GL_QUADS)
        # Front face (z+)
        glColor3f(1.0, 0.0, 0.0)  # red
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # Back face (z-)
        glColor3f(0.0, 1.0, 0.0)  # green
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # Left face (x-)
        glColor3f(0.0, 0.0, 1.0)  # blue
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # Right face (x+)
        glColor3f(1.0, 1.0, 0.0)  # yellow
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Top face (y+)
        glColor3f(1.0, 0.0, 1.0)  # magenta
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Bottom face (y-)
        glColor3f(0.0, 1.0, 1.0)  # cyan
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glEnd()

        glPopMatrix()

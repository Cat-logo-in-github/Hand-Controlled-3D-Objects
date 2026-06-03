from OpenGL.GL import *
from .grabbable_object import GrabbableObject

class Cuboid(GrabbableObject):
    def __init__(self, width=1.0, height=1.0, depth=1.0, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.depth = depth

    def render(self):
        glPushMatrix()
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale[0] * self.width, self.scale[1] * self.height, self.scale[2] * self.depth)

        glBegin(GL_QUADS)
        # Front face
        glColor3f(0.8, 0.2, 0.2)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # Back face
        glColor3f(0.2, 0.8, 0.2)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # Left face
        glColor3f(0.2, 0.2, 0.8)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # Right face
        glColor3f(0.8, 0.8, 0.2)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Top face
        glColor3f(0.8, 0.2, 0.8)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Bottom face
        glColor3f(0.2, 0.8, 0.8)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glEnd()
        glPopMatrix()

"""Data structures."""

from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from numpy.typing import NDArray


class Geom2D(abc.ABC):
    """A 2D shape that contains some points."""

    @abc.abstractmethod
    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        """Plot the shape on a given pyplot axis."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """Checks if a point is contained in the shape."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Samples a random point inside the 2D shape."""
        raise NotImplementedError("Override me!")

    def intersects(self, other: Geom2D) -> bool:
        """Checks if this shape intersects with another one."""
        return geom2ds_intersect(self, other)


@dataclass(frozen=True)
class LineSegment(Geom2D):
    """A helper class for visualizing and collision checking line segments."""

    x1: float
    y1: float
    x2: float
    y2: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        ax.plot([self.x1, self.x2], [self.y1, self.y2], **kwargs)

    def contains_point(self, x: float, y: float) -> bool:
        # https://stackoverflow.com/questions/328107
        a = (self.x1, self.y1)
        b = (self.x2, self.y2)
        c = (x, y)
        # Need to use an epsilon for numerical stability. But we are checking
        # if the distance from a to b is (approximately) equal to the distance
        # from a to c and the distance from c to b.
        eps = 1e-6

        def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
            return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

        return -eps < _dist(a, c) + _dist(c, b) - _dist(a, b) < eps

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        line_slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        y_intercept = self.y2 - (line_slope * self.x2)
        random_x_point = rng.uniform(self.x1, self.x2)
        random_y_point_on_line = line_slope * random_x_point + y_intercept
        assert self.contains_point(random_x_point, random_y_point_on_line)
        return (random_x_point, random_y_point_on_line)


@dataclass(frozen=True)
class Circle(Geom2D):
    """A helper class for visualizing and collision checking circles."""

    x: float
    y: float
    radius: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        patch = patches.Circle((self.x, self.y), self.radius, **kwargs)
        ax.add_patch(patch)

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius**2

    def contains_circle(self, other_circle: Circle) -> bool:
        """Check whether this circle wholly contains another one."""
        dist_between_centers = np.sqrt(
            (other_circle.x - self.x) ** 2 + (other_circle.y - self.y) ** 2
        )
        return (dist_between_centers + other_circle.radius) <= self.radius

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        rand_mag = rng.uniform(0, self.radius)
        rand_theta = rng.uniform(0, 2 * np.pi)
        x_point = self.x + rand_mag * np.cos(rand_theta)
        y_point = self.y + rand_mag * np.sin(rand_theta)
        assert self.contains_point(x_point, y_point)
        return (x_point, y_point)


@dataclass(frozen=True)
class Triangle(Geom2D):
    """A helper class for visualizing and collision checking triangles."""

    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        patch = patches.Polygon(
            [[self.x1, self.y1], [self.x2, self.y2], [self.x3, self.y3]], **kwargs
        )
        ax.add_patch(patch)

    def __post_init__(self) -> None:
        dist1 = np.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)
        dist2 = np.sqrt((self.x2 - self.x3) ** 2 + (self.y2 - self.y3) ** 2)
        dist3 = np.sqrt((self.x3 - self.x1) ** 2 + (self.y3 - self.y1) ** 2)
        dists = sorted([dist1, dist2, dist3])
        assert dists[0] + dists[1] >= dists[2]
        if dists[0] + dists[1] == dists[2]:
            raise ValueError("Degenerate triangle!")

    def contains_point(self, x: float, y: float) -> bool:
        # Adapted from https://stackoverflow.com/questions/2049582/.
        sign1 = (
            (x - self.x2) * (self.y1 - self.y2) - (self.x1 - self.x2) * (y - self.y2)
        ) > 0
        sign2 = (
            (x - self.x3) * (self.y2 - self.y3) - (self.x2 - self.x3) * (y - self.y3)
        ) > 0
        sign3 = (
            (x - self.x1) * (self.y3 - self.y1) - (self.x3 - self.x1) * (y - self.y1)
        ) > 0
        has_neg = (not sign1) or (not sign2) or (not sign3)
        has_pos = sign1 or sign2 or sign3
        return not has_neg or not has_pos

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        a = np.array([self.x2 - self.x1, self.y2 - self.y1])
        b = np.array([self.x3 - self.x1, self.y3 - self.y1])
        u1 = rng.uniform(0, 1)
        u2 = rng.uniform(0, 1)
        if u1 + u2 > 1.0:
            u1 = 1 - u1
            u2 = 1 - u2
        point_in_triangle = (u1 * a + u2 * b) + np.array([self.x1, self.y1])
        assert self.contains_point(point_in_triangle[0], point_in_triangle[1])
        return (point_in_triangle[0], point_in_triangle[1])


@dataclass(frozen=True)
class Rectangle(Geom2D):
    """A helper class for visualizing and collision checking rectangles.

    Following the convention in plt.Rectangle, the origin is at the
    bottom left corner, and rotation is anti-clockwise about that point.

    Unlike plt.Rectangle, the angle is in radians.
    """

    x: float
    y: float
    width: float
    height: float
    theta: float  # in radians, between -np.pi and np.pi

    def __post_init__(self) -> None:
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @staticmethod
    def from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        rotation_about_center: float,
    ) -> Rectangle:
        """Create a rectangle given an (x, y) for the center, with theta
        rotating about that center point."""
        x = center_x - width / 2
        y = center_y - height / 2
        norm_rect = Rectangle(x, y, width, height, 0.0)
        assert np.isclose(norm_rect.center[0], center_x)
        assert np.isclose(norm_rect.center[1], center_y)
        return norm_rect.rotate_about_point(center_x, center_y, rotation_about_center)

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices for the rectangle."""
        scale_matrix = np.array(
            [
                [self.width, 0],
                [0, self.height],
            ]
        )
        translate_vector = np.array([self.x, self.y])
        vertices = np.array(
            [
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 0),
            ]
        )
        vertices = vertices @ scale_matrix.T
        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        # Convert to a list of tuples. Slightly complicated to appease both
        # type checking and linting.
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the four line segments for the rectangle."""
        vs = list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))
        line_segments = []
        for (x1, y1), (x2, y2) in vs:
            line_segments.append(LineSegment(x1, y1, x2, y2))
        return line_segments

    @functools.cached_property
    def center(self) -> Tuple[float, float]:
        """Get the point at the center of the rectangle."""
        x, y = np.mean(self.vertices, axis=0)
        return (x, y)

    @functools.cached_property
    def circumscribed_circle(self) -> Circle:
        """Returns x, y, radius."""
        x, y = self.center
        radius = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        return Circle(x, y, radius)

    def contains_point(self, x: float, y: float) -> bool:
        # First invert translation, then invert rotation.
        rx, ry = np.array([x - self.x, y - self.y]) @ self.inverse_rotation_matrix.T
        return 0 <= rx <= self.width and 0 <= ry <= self.height

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        rand_width = rng.uniform(0, self.width)
        rand_height = rng.uniform(0, self.height)
        # First rotate, then translate.
        rx, ry = np.array([rand_width, rand_height]) @ self.rotation_matrix.T
        x = rx + self.x
        y = ry + self.y
        assert self.contains_point(x, y)
        return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> Rectangle:
        """Create a new rectangle that is this rectangle, but rotated CCW by
        the given rotation (in radians), relative to the (x, y) origin.

        Rotates the vertices first, then uses them to recompute the new
        theta.
        """
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        # Translate the vertices so that they become the "origin".
        vertices = vertices - origin
        # Rotate.
        rotate_matrix = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
        )
        vertices = vertices @ rotate_matrix.T
        # Translate the vertices back.
        vertices = vertices + origin
        # Recompute theta.
        (lx, ly), _, _, (rx, ry) = vertices
        theta = np.arctan2(ry - ly, rx - lx)
        rect = Rectangle(lx, ly, self.width, self.height, theta)
        # assert np.allclose(rect.vertices, vertices)
        return rect

    def scale_about_center(self, width_scale: float, height_scale: float) -> Rectangle:
        """Create a new rectangle that is this rectangle, shrunken or stretched
        about the center of this rectangle."""
        new_height = self.height * height_scale
        new_width = self.width * width_scale
        return Rectangle.from_center(
            self.center[0], self.center[1], new_width, new_height, self.theta
        )

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        angle = self.theta * 180 / np.pi
        patch = patches.Rectangle(
            (self.x, self.y), self.width, self.height, angle=angle, **kwargs
        )
        ax.add_patch(patch)


@dataclass(frozen=True)
class RTrapezoid(Geom2D):
    """A helper class for representing a right-angled trapezoid.

    The trapezoid has:
    - A long side of length l (bottom side)
    - A height h (perpendicular to the long side)
    - A short side of length l-h (top side, parallel to bottom)
    - The right angle is at the origin (x, y) where the long side meets the left edge

    Vertices in unrotated configuration:
        (x, y+h) ------- (x+l-h, y+h)  [short side: l-h]
           |                /
           |              /   [slanted side]
           |            /
        (x, y) ------- (x+l, y)  [long side: l]
    """

    x: float  # Origin at the right angle
    y: float
    l: float  # Length of the long side
    h: float  # Height of the trapezoid
    theta: float  # Rotation angle in radians

    def __post_init__(self) -> None:
        """Validate the trapezoid parameters."""
        if self.l <= 0:
            raise ValueError("Length l must be positive.")
        if self.h <= 0:
            raise ValueError("Height h must be positive.")
        if self.h >= self.l:
            raise ValueError("Height h must be less than length l (l-h > 0).")
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices of the right-angled trapezoid."""
        translate_vector = np.array([self.x, self.y])
        # Define vertices in unrotated frame (CCW order starting from origin)
        vertices = np.array(
            [
                (0, 0),  # Bottom left (origin, right angle)
                (self.l, 0),  # Bottom right
                (self.l - self.h, self.h),  # Top right
                (0, self.h),  # Top left
            ]
        )
        # Apply rotation then translation
        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the four line segments for the trapezoid."""
        vs = list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))
        line_segments = []
        for (x1, y1), (x2, y2) in vs:
            line_segments.append(LineSegment(x1, y1, x2, y2))
        return line_segments

    @functools.cached_property
    def center(self) -> Tuple[float, float]:
        """Get the centroid of the trapezoid."""
        x, y = np.mean(self.vertices, axis=0)
        return (x, y)

    @functools.cached_property
    def area(self) -> float:
        """Calculate the area of the trapezoid: (l + (l-h)) * h / 2."""
        return (self.l + (self.l - self.h)) * self.h / 2

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the trapezoid using cross products."""
        # Transform point to local coordinates
        rx, ry = np.array([x - self.x, y - self.y]) @ self.inverse_rotation_matrix.T

        # Check basic bounds first
        if not (0 <= rx <= self.l and 0 <= ry <= self.h):
            return False

        # Check if point is below the slanted line
        # The slanted line goes from (l, 0) to (l-h, h)
        # Equation: For x from l-h to l, y_max = h * (l - x) / h = l - x
        # So we need: y <= l - x (in the region where x > l-h)
        if rx > self.l - self.h:
            # In the region with the slant
            y_max_at_rx = self.h * (self.l - rx) / self.h
            if ry > y_max_at_rx:
                return False

        return True

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Sample a random point inside the trapezoid.

        Uses rejection sampling within the bounding rectangle.
        """
        while True:
            # Sample from bounding rectangle
            rx = rng.uniform(0, self.l)
            ry = rng.uniform(0, self.h)

            # Check if point is valid (below slanted edge)
            if rx <= self.l - self.h or ry <= self.h * (self.l - rx) / self.h:
                # Rotate and translate
                rotated = np.array([rx, ry]) @ self.rotation_matrix.T
                x = rotated[0] + self.x
                y = rotated[1] + self.y
                assert self.contains_point(x, y)
                return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> RTrapezoid:
        """Create a new trapezoid rotated CCW by rot radians about (x, y)."""
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        # Translate to origin
        vertices = vertices - origin
        # Rotate
        rotate_matrix = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
        )
        vertices = vertices @ rotate_matrix.T
        # Translate back
        vertices = vertices + origin

        # New origin is the first vertex (bottom-left)
        new_x, new_y = vertices[0]
        # Compute new theta from bottom edge direction
        bottom_edge = vertices[1] - vertices[0]
        new_theta = np.arctan2(bottom_edge[1], bottom_edge[0])

        return RTrapezoid(new_x, new_y, self.l, self.h, new_theta)

    def scale_about_center(self, scale: float) -> RTrapezoid:
        """Scale the trapezoid uniformly about its center."""
        center_x, center_y = self.center
        new_l = self.l * scale
        new_h = self.h * scale

        # Create an un-rotated trapezoid with new dimensions at origin
        temp_trap = RTrapezoid(0, 0, new_l, new_h, 0)
        # Get its centroid in local (unrotated) coordinates
        temp_center_local = temp_trap.center

        # The new origin should be positioned such that when rotated,
        # the centroid ends up at (center_x, center_y)
        # Inverse transform: go from desired center to origin position
        center_vector = np.array([center_x, center_y])
        local_center_vector = np.array(temp_center_local)

        # Rotate local center to world coordinates
        rotated_center = local_center_vector @ self.rotation_matrix.T

        # New origin = desired_center - rotated_local_center
        new_origin = center_vector - rotated_center

        return RTrapezoid(new_origin[0], new_origin[1], new_l, new_h, self.theta)

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        """Plot the trapezoid as a polygon patch."""
        patch = patches.Polygon(self.vertices, **kwargs)
        ax.add_patch(patch)


@dataclass(frozen=True)
class Lobject(Geom2D):
    """A helper class for representing a L-shaped object for visualizing and
    collision checking.

    The hook body pose is defined as follows:
    --------|
            |
    The hook looks like above when theta = 0.0,
    right is x positive, up is y positive.
    The pose refers to the very right top vertex of the hook.
    The geometry is defined by two (non-overlapping) rectanges
    The "horizonal" one has width=hook_shape[1]-hook_shape[0],
    and height=hook_shape[0]
    The "vertical" one has width=hook_shape[0 and height=hook_shape[2]
    """

    x: float
    y: float
    width: float
    lengths: tuple[float, float]
    theta: float  # Rotation angle in radians with respect to (x,y)

    def __post_init__(self):
        # Ensure that the lengths are positive
        if any(length <= 0 for length in self.lengths):
            raise ValueError("Lengths must be positive.")
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the eight vertices of the L-object.

        The last two vertices are for plotting convenience.
        """
        w, l1, l2 = self.width, self.lengths[0], self.lengths[1]
        translate_vector = np.array([self.x, self.y])
        vertices = np.array(
            [
                (0, 0),
                (-l1, 0),
                (-l1, -w),
                (-w, -w),
                (-w, -l2),
                (0, -l2),
                (0, -w),
                (-w, 0),
            ]
        )

        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        # Convert to a list of tuples. Slightly complicated to appease both
        # type checking and linting.
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the six line segments of the L-object."""
        v = self.vertices
        return [
            LineSegment(v[0][0], v[0][1], v[1][0], v[1][1]),
            LineSegment(v[1][0], v[1][1], v[2][0], v[2][1]),
            LineSegment(v[2][0], v[2][1], v[3][0], v[3][1]),
            LineSegment(v[3][0], v[3][1], v[4][0], v[4][1]),
            LineSegment(v[4][0], v[4][1], v[5][0], v[5][1]),
            LineSegment(v[5][0], v[5][1], v[0][0], v[0][1]),
        ]

    def contains_point(self, x: float, y: float) -> bool:
        # First invert translation, then invert rotation.
        rx, ry = np.array([x - self.x, y - self.y]) @ self.inverse_rotation_matrix.T
        # Check if the point is within the bounds of the L-object.
        return (-self.lengths[0] <= rx <= 0 and -self.width <= ry <= 0) or (
            -self.width <= rx <= 0 and -self.lengths[1] <= ry <= 0
        )

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        # Sample a random point within the bounds of the L-object.

        side_length = rng.choice([self.lengths[0], self.lengths[1]])
        if side_length == self.lengths[0]:
            rx = -rng.uniform(0, self.lengths[0])
            ry = -rng.uniform(0, self.width)
        else:
            rx = -rng.uniform(0, self.width)
            ry = -rng.uniform(0, self.lengths[1])

        rx, ry = np.array([rx, ry]) @ self.rotation_matrix.T
        x = rx + self.x
        y = ry + self.y

        return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> Lobject:
        """Create a new L-object that is this L-object, but rotated CCW by the
        given rotation (in radians), relative to the (x, y) origin.

        Rotates the vertices first, then uses them to recompute the new
        theta.
        """
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        # Translate the vertices so that they become the "origin".
        vertices = vertices - origin
        # Rotate.
        rotate_matrix = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
        )
        vertices = vertices @ rotate_matrix.T

        # Translate the vertices back.
        vertices = vertices + origin

        center_x, center_y = vertices[0, 0], vertices[0, 1]
        width = np.linalg.norm(vertices[1] - vertices[2]).item()

        vector_side1 = vertices[1] - vertices[0]
        vector_side2 = vertices[5] - vertices[0]
        length_side1 = np.linalg.norm(vector_side1).item()
        length_side2 = np.linalg.norm(vector_side2).item()

        # Compute new_theta based on dot product of vector_side2 and x-axis
        new_theta = np.arctan2(-vector_side1[1], -vector_side1[0])

        return Lobject(
            center_x, center_y, width, (length_side1, length_side2), new_theta
        )

    def scale_about_center(self, width_scale: float, length_scale: float) -> Lobject:
        """Scale the L-object about its center."""
        # Compute the new dimensions.
        new_length_side1 = self.lengths[0] * width_scale
        new_length_side2 = self.lengths[1] * length_scale
        new_width = self.width * width_scale
        # Create a new L-object with the new dimensions.
        return Lobject(
            self.x, self.y, new_width, (new_length_side1, new_length_side2), self.theta
        )

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        vertices = self.vertices

        rectangle1_vertices = np.array(
            [
                [vertices[0][0], vertices[0][1]],
                [vertices[1][0], vertices[1][1]],
                [vertices[2][0], vertices[2][1]],
                [vertices[6][0], vertices[6][1]],
            ]
        )
        rectangle2_vertices = np.array(
            [
                [vertices[4][0], vertices[4][1]],
                [vertices[5][0], vertices[5][1]],
                [vertices[0][0], vertices[0][1]],
                [vertices[7][0], vertices[7][1]],
            ]
        )

        # Create rectangle patches
        rect1_patch = plt.Polygon(rectangle1_vertices, closed=True, fill=True, **kwargs)
        rect2_patch = plt.Polygon(rectangle2_vertices, closed=True, fill=True, **kwargs)
        ax.add_patch(rect1_patch)
        ax.add_patch(rect2_patch)


@dataclass(frozen=True)
class Tobject(Geom2D):
    """A helper class for representing a T-shaped object for visualizing and
    collision checking.

    The T-shaped body pose is defined as follows:
    -----
      |
      |
    The T looks like above when theta = 0.0,
    right is x positive, up is y positive.
    The pose refers to the center top of the horizontal bar.
    The geometry is defined by two (non-overlapping) rectangles:
    - The "horizontal" one has width=length_horizontal and height=width
    - The "vertical" one has width=width and height=length_vertical
    """

    x: float
    y: float
    width: float
    length_horizontal: float
    length_vertical: float
    theta: float  # Rotation angle in radians with respect to (x,y)

    def __post_init__(self):
        # Ensure that the lengths are positive
        if self.width <= 0 or self.length_horizontal <= 0 or self.length_vertical <= 0:
            raise ValueError("Width and lengths must be positive.")
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the eight vertices of the T-object.

        The vertices trace the outline of the T-shape.
        """
        w = self.width
        lh = self.length_horizontal
        lv = self.length_vertical
        translate_vector = np.array([self.x, self.y])

        # Define vertices in local frame (before rotation)
        # Starting from center top, going clockwise
        vertices = np.array(
            [
                (-lh / 2, 0),  # top left of horizontal bar
                (-lh / 2, -w),  # bottom left of horizontal bar
                (-w / 2, -w),  # top left of vertical bar
                (-w / 2, -w - lv),  # bottom left of vertical bar
                (w / 2, -w - lv),  # bottom right of vertical bar
                (w / 2, -w),  # top right of vertical bar
                (lh / 2, -w),  # bottom right of horizontal bar
                (lh / 2, 0),  # top right of horizontal bar
            ]
        )

        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        # Convert to a list of tuples
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the eight line segments of the T-object."""
        v = self.vertices
        return [
            LineSegment(v[i][0], v[i][1], v[(i + 1) % 8][0], v[(i + 1) % 8][1])
            for i in range(8)
        ]

    def contains_point(self, x: float, y: float) -> bool:
        # First invert translation, then invert rotation.
        rx, ry = np.array([x - self.x, y - self.y]) @ self.inverse_rotation_matrix.T
        # Check if the point is within the bounds of the T-object.
        # Horizontal bar: from -lh/2 to lh/2 in x, from 0 to -w in y
        in_horizontal = (
            -self.length_horizontal / 2 <= rx <= self.length_horizontal / 2
            and -self.width <= ry <= 0
        )
        # Vertical bar: from -w/2 to w/2 in x, from -w to -w-lv in y
        in_vertical = (
            -self.width / 2 <= rx <= self.width / 2
            and -self.width - self.length_vertical <= ry <= -self.width
        )
        return in_horizontal or in_vertical

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        # Sample a random point within the bounds of the T-object.
        # Choose randomly between horizontal and vertical bars
        horizontal_area = self.length_horizontal * self.width
        vertical_area = self.length_vertical * self.width
        total_area = horizontal_area + vertical_area

        if rng.uniform(0, total_area) < horizontal_area:
            # Sample from horizontal bar
            rx = rng.uniform(-self.length_horizontal / 2, self.length_horizontal / 2)
            ry = rng.uniform(-self.width, 0)
        else:
            # Sample from vertical bar
            rx = rng.uniform(-self.width / 2, self.width / 2)
            ry = rng.uniform(-self.width - self.length_vertical, -self.width)

        rx, ry = np.array([rx, ry]) @ self.rotation_matrix.T
        x = rx + self.x
        y = ry + self.y

        return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> Tobject:
        """Create a new T-object that is this T-object, but rotated CCW by the
        given rotation (in radians), relative to the (x, y) origin.

        Rotates the vertices first, then uses them to recompute the new
        theta.
        """
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        # Translate the vertices so that the origin becomes the center
        vertices = vertices - origin
        # Rotate
        rotate_matrix = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
        )
        vertices = vertices @ rotate_matrix.T

        # Translate the vertices back
        vertices = vertices + origin

        # New center is at vertex 0 (or average of vertices 0 and 7)
        center_x, center_y = (vertices[0, 0] + vertices[7, 0]) / 2, vertices[0, 1]

        # Compute dimensions
        width = np.linalg.norm(vertices[1] - vertices[0]).item()
        length_horizontal = np.linalg.norm(vertices[7] - vertices[0]).item()

        # Compute length_vertical from bottom vertices
        length_vertical = np.linalg.norm(vertices[3] - vertices[2]).item()

        # Compute new_theta based on the direction of the top edge
        vector_top = vertices[7] - vertices[0]
        new_theta = np.arctan2(vector_top[1], vector_top[0])

        return Tobject(
            center_x, center_y, width, length_horizontal, length_vertical, new_theta
        )

    def scale_about_center(self, width_scale: float, length_scale: float) -> Tobject:
        """Scale the T-object about its center."""
        new_width = self.width * width_scale
        new_length_horizontal = self.length_horizontal * length_scale
        new_length_vertical = self.length_vertical * length_scale
        return Tobject(
            self.x,
            self.y,
            new_width,
            new_length_horizontal,
            new_length_vertical,
            self.theta,
        )

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        vertices = self.vertices

        # Horizontal bar vertices (indices 0, 1, 6, 7)
        horizontal_vertices = np.array(
            [
                [vertices[0][0], vertices[0][1]],
                [vertices[1][0], vertices[1][1]],
                [vertices[6][0], vertices[6][1]],
                [vertices[7][0], vertices[7][1]],
            ]
        )

        # Vertical bar vertices (indices 2, 3, 4, 5)
        vertical_vertices = np.array(
            [
                [vertices[2][0], vertices[2][1]],
                [vertices[3][0], vertices[3][1]],
                [vertices[4][0], vertices[4][1]],
                [vertices[5][0], vertices[5][1]],
            ]
        )

        # Create rectangle patches
        horizontal_patch = plt.Polygon(
            horizontal_vertices, closed=True, fill=True, **kwargs
        )
        vertical_patch = plt.Polygon(
            vertical_vertices, closed=True, fill=True, **kwargs
        )
        ax.add_patch(horizontal_patch)
        ax.add_patch(vertical_patch)


def line_segments_intersect(seg1: LineSegment, seg2: LineSegment) -> bool:
    """Checks if two line segments intersect.

    This method, which works by checking relative orientation, allows
    for collinearity, and only checks if each segment straddles the line
    containing the other.
    """

    def _subtract(
        a: Tuple[float, float], b: Tuple[float, float]
    ) -> Tuple[float, float]:
        x1, y1 = a
        x2, y2 = b
        return (x1 - x2), (y1 - y2)

    def _cross_product(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        x1, y1 = b
        x2, y2 = a
        return x1 * y2 - x2 * y1

    def _direction(
        a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> float:
        return _cross_product(_subtract(a, c), _subtract(a, b))

    p1 = (seg1.x1, seg1.y1)
    p2 = (seg1.x2, seg1.y2)
    p3 = (seg2.x1, seg2.y1)
    p4 = (seg2.x2, seg2.y2)
    d1 = _direction(p3, p4, p1)
    d2 = _direction(p3, p4, p2)
    d3 = _direction(p1, p2, p3)
    d4 = _direction(p1, p2, p4)

    return ((d2 < 0 < d1) or (d1 < 0 < d2)) and ((d4 < 0 < d3) or (d3 < 0 < d4))


def circles_intersect(circ1: Circle, circ2: Circle) -> bool:
    """Checks if two circles intersect."""
    x1, y1, r1 = circ1.x, circ1.y, circ1.radius
    x2, y2, r2 = circ2.x, circ2.y, circ2.radius
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r1 + r2) ** 2


def rectangles_intersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Checks if two rectangles intersect."""
    # Optimization: if the circumscribed circles don't intersect, then
    # the rectangles also don't intersect.
    if not circles_intersect(rect1.circumscribed_circle, rect2.circumscribed_circle):
        return False
    # Case 1: line segments intersect.
    if any(
        line_segments_intersect(seg1, seg2)
        for seg1 in rect1.line_segments
        for seg2 in rect2.line_segments
    ):
        return True
    # Case 2: rect1 inside rect2.
    if rect1.contains_point(rect2.center[0], rect2.center[1]):
        return True
    # Case 3: rect2 inside rect1.
    if rect2.contains_point(rect1.center[0], rect1.center[1]):
        return True
    # Not intersecting.
    return False


def line_segment_intersects_circle(
    seg: LineSegment,
    circ: Circle,
) -> bool:
    """Checks if a line segment intersects a circle."""
    # First check if the end points of the segment are in the circle.
    if circ.contains_point(seg.x1, seg.y1):
        return True
    if circ.contains_point(seg.x2, seg.y2):
        return True
    # Project the circle radius onto the extended line.
    c = (circ.x, circ.y)
    # Project (a, c) onto (a, b).
    a = (seg.x1, seg.y1)
    b = (seg.x2, seg.y2)
    ba = np.subtract(b, a)
    ca = np.subtract(c, a)
    da = ba * np.dot(ca, ba) / np.dot(ba, ba)
    # The point on the extended line that is the closest to the center.
    dx, dy = (a[0] + da[0], a[1] + da[1])
    # Check if the point is on the line. If it's not, there is no intersection,
    # because we already checked that the circle does not contain the end
    # points of the line segment.
    if not seg.contains_point(dx, dy):
        return False
    # So d is on the segment. Check if it's in the circle.
    return circ.contains_point(dx, dy)


def line_segment_intersects_rectangle(seg: LineSegment, rect: Rectangle) -> bool:
    """Checks if a line segment intersects a rectangle."""
    # Case 1: one of the end points of the segment is in the rectangle.
    if rect.contains_point(seg.x1, seg.y1) or rect.contains_point(seg.x2, seg.y2):
        return True
    # Case 2: the segment intersects with one of the rectangle sides.
    return any(line_segments_intersect(s, seg) for s in rect.line_segments)


def rectangle_intersects_circle(rect: Rectangle, circ: Circle) -> bool:
    """Checks if a rectangle intersects a circle."""
    # Optimization: if the circumscribed circle of the rectangle doesn't
    # intersect with the circle, then there can't be an intersection.
    if not circles_intersect(rect.circumscribed_circle, circ):
        return False
    # Case 1: the circle's center is in the rectangle.
    if rect.contains_point(circ.x, circ.y):
        return True
    # Case 2: one of the sides of the rectangle intersects the circle.
    for seg in rect.line_segments:
        if line_segment_intersects_circle(seg, circ):
            return True
    return False


def rtrapezoids_intersect(trap1: RTrapezoid, trap2: RTrapezoid) -> bool:
    """Checks if two right-angled trapezoids intersect."""
    # Case 1: line segments intersect.
    if any(
        line_segments_intersect(seg1, seg2)
        for seg1 in trap1.line_segments
        for seg2 in trap2.line_segments
    ):
        return True
    # Case 2: trap1 inside trap2.
    if trap1.contains_point(trap2.center[0], trap2.center[1]):
        return True
    # Case 3: trap2 inside trap1.
    if trap2.contains_point(trap1.center[0], trap1.center[1]):
        return True
    # Not intersecting.
    return False


def line_segment_intersects_rtrapezoid(seg: LineSegment, trap: RTrapezoid) -> bool:
    """Checks if a line segment intersects a right-angled trapezoid."""
    # Case 1: one of the end points of the segment is in the trapezoid.
    if trap.contains_point(seg.x1, seg.y1) or trap.contains_point(seg.x2, seg.y2):
        return True
    # Case 2: the segment intersects with one of the trapezoid sides.
    return any(line_segments_intersect(s, seg) for s in trap.line_segments)


def rtrapezoid_intersects_circle(trap: RTrapezoid, circ: Circle) -> bool:
    """Checks if a right-angled trapezoid intersects a circle."""
    # Case 1: the circle's center is in the trapezoid.
    if trap.contains_point(circ.x, circ.y):
        return True
    # Case 2: one of the sides of the trapezoid intersects the circle.
    for seg in trap.line_segments:
        if line_segment_intersects_circle(seg, circ):
            return True
    return False


def rtrapezoid_intersects_rectangle(trap: RTrapezoid, rect: Rectangle) -> bool:
    """Checks if a right-angled trapezoid intersects a rectangle."""
    # Case 1: line segments intersect.
    if any(
        line_segments_intersect(seg1, seg2)
        for seg1 in trap.line_segments
        for seg2 in rect.line_segments
    ):
        return True
    # Case 2: trap inside rect.
    if trap.contains_point(rect.center[0], rect.center[1]):
        return True
    # Case 3: rect inside trap.
    if rect.contains_point(trap.center[0], trap.center[1]):
        return True
    # Not intersecting.
    return False


def line_segment_intersects_triangle(seg: LineSegment, tri: Triangle) -> bool:
    """Checks if a line segment intersects a triangle."""
    # Case 1: one of the end points of the segment is in the triangle.
    if tri.contains_point(seg.x1, seg.y1) or tri.contains_point(seg.x2, seg.y2):
        return True
    # Case 2: the segment intersects with one of the triangle sides.
    tri_seg1 = LineSegment(tri.x1, tri.y1, tri.x2, tri.y2)
    tri_seg2 = LineSegment(tri.x2, tri.y2, tri.x3, tri.y3)
    tri_seg3 = LineSegment(tri.x3, tri.y3, tri.x1, tri.y1)
    return any(
        line_segments_intersect(seg, tri_seg)
        for tri_seg in [tri_seg1, tri_seg2, tri_seg3]
    )


def triangle_intersects_circle(tri: Triangle, circ: Circle) -> bool:
    """Checks if a triangle intersects a circle."""
    # Case 1: the circle's center is in the triangle.
    if tri.contains_point(circ.x, circ.y):
        return True
    # Case 2: any vertex of the triangle is inside the circle.
    if circ.contains_point(tri.x1, tri.y1):
        return True
    if circ.contains_point(tri.x2, tri.y2):
        return True
    if circ.contains_point(tri.x3, tri.y3):
        return True
    # Case 3: one of the sides of the triangle intersects the circle.
    tri_seg1 = LineSegment(tri.x1, tri.y1, tri.x2, tri.y2)
    tri_seg2 = LineSegment(tri.x2, tri.y2, tri.x3, tri.y3)
    tri_seg3 = LineSegment(tri.x3, tri.y3, tri.x1, tri.y1)
    for seg in [tri_seg1, tri_seg2, tri_seg3]:
        if line_segment_intersects_circle(seg, circ):
            return True
    return False


def triangle_intersects_rectangle(tri: Triangle, rect: Rectangle) -> bool:
    """Checks if a triangle intersects a rectangle."""
    # Case 1: any vertex of the triangle is inside the rectangle.
    if rect.contains_point(tri.x1, tri.y1):
        return True
    if rect.contains_point(tri.x2, tri.y2):
        return True
    if rect.contains_point(tri.x3, tri.y3):
        return True
    # Case 2: any vertex of the rectangle is inside the triangle.
    if any(tri.contains_point(vx, vy) for vx, vy in rect.vertices):
        return True
    # Case 3: any edge of the triangle intersects any edge of the rectangle.
    tri_seg1 = LineSegment(tri.x1, tri.y1, tri.x2, tri.y2)
    tri_seg2 = LineSegment(tri.x2, tri.y2, tri.x3, tri.y3)
    tri_seg3 = LineSegment(tri.x3, tri.y3, tri.x1, tri.y1)
    for tri_seg in [tri_seg1, tri_seg2, tri_seg3]:
        for rect_seg in rect.line_segments:
            if line_segments_intersect(tri_seg, rect_seg):
                return True
    return False


def triangles_intersect(tri1: Triangle, tri2: Triangle) -> bool:
    """Checks if two triangles intersect."""
    # Case 1: any vertex of tri1 is inside tri2.
    if tri2.contains_point(tri1.x1, tri1.y1):
        return True
    if tri2.contains_point(tri1.x2, tri1.y2):
        return True
    if tri2.contains_point(tri1.x3, tri1.y3):
        return True
    # Case 2: any vertex of tri2 is inside tri1.
    if tri1.contains_point(tri2.x1, tri2.y1):
        return True
    if tri1.contains_point(tri2.x2, tri2.y2):
        return True
    if tri1.contains_point(tri2.x3, tri2.y3):
        return True
    # Case 3: any edge of tri1 intersects any edge of tri2.
    tri1_seg1 = LineSegment(tri1.x1, tri1.y1, tri1.x2, tri1.y2)
    tri1_seg2 = LineSegment(tri1.x2, tri1.y2, tri1.x3, tri1.y3)
    tri1_seg3 = LineSegment(tri1.x3, tri1.y3, tri1.x1, tri1.y1)
    tri2_seg1 = LineSegment(tri2.x1, tri2.y1, tri2.x2, tri2.y2)
    tri2_seg2 = LineSegment(tri2.x2, tri2.y2, tri2.x3, tri2.y3)
    tri2_seg3 = LineSegment(tri2.x3, tri2.y3, tri2.x1, tri2.y1)
    for seg1 in [tri1_seg1, tri1_seg2, tri1_seg3]:
        for seg2 in [tri2_seg1, tri2_seg2, tri2_seg3]:
            if line_segments_intersect(seg1, seg2):
                return True
    return False


def triangle_intersects_rtrapezoid(tri: Triangle, trap: RTrapezoid) -> bool:
    """Checks if a triangle intersects a right-angled trapezoid."""
    # Case 1: any vertex of the triangle is inside the trapezoid.
    if trap.contains_point(tri.x1, tri.y1):
        return True
    if trap.contains_point(tri.x2, tri.y2):
        return True
    if trap.contains_point(tri.x3, tri.y3):
        return True
    # Case 2: any vertex of the trapezoid is inside the triangle.
    if any(tri.contains_point(vx, vy) for vx, vy in trap.vertices):
        return True
    # Case 3: any edge of the triangle intersects any edge of the trapezoid.
    tri_seg1 = LineSegment(tri.x1, tri.y1, tri.x2, tri.y2)
    tri_seg2 = LineSegment(tri.x2, tri.y2, tri.x3, tri.y3)
    tri_seg3 = LineSegment(tri.x3, tri.y3, tri.x1, tri.y1)
    for tri_seg in [tri_seg1, tri_seg2, tri_seg3]:
        for trap_seg in trap.line_segments:
            if line_segments_intersect(tri_seg, trap_seg):
                return True
    return False


def triangle_intersects_lobject(tri: Triangle, lobj: Lobject) -> bool:
    """Checks if a triangle intersects an L-object."""
    # Case 1: any vertex of the triangle is inside the L-object.
    if lobj.contains_point(tri.x1, tri.y1):
        return True
    if lobj.contains_point(tri.x2, tri.y2):
        return True
    if lobj.contains_point(tri.x3, tri.y3):
        return True
    # Case 2: any vertex of the L-object is inside the triangle.
    if any(tri.contains_point(vx, vy) for vx, vy in lobj.vertices):
        return True
    # Case 3: any edge of the triangle intersects any edge of the L-object.
    tri_seg1 = LineSegment(tri.x1, tri.y1, tri.x2, tri.y2)
    tri_seg2 = LineSegment(tri.x2, tri.y2, tri.x3, tri.y3)
    tri_seg3 = LineSegment(tri.x3, tri.y3, tri.x1, tri.y1)
    for tri_seg in [tri_seg1, tri_seg2, tri_seg3]:
        for lobj_seg in lobj.line_segments:
            if line_segments_intersect(tri_seg, lobj_seg):
                return True
    return False


def geom2ds_intersect(geom1: Geom2D, geom2: Geom2D) -> bool:
    """Check if two 2D bodies intersect."""
    if isinstance(geom1, LineSegment) and isinstance(geom2, LineSegment):
        return line_segments_intersect(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Circle):
        return line_segment_intersects_circle(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Rectangle):
        return line_segment_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, RTrapezoid):
        return line_segment_intersects_rtrapezoid(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Triangle):
        return line_segment_intersects_triangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_rectangle(geom2, geom1)
    if isinstance(geom1, RTrapezoid) and isinstance(geom2, LineSegment):
        return line_segment_intersects_rtrapezoid(geom2, geom1)
    if isinstance(geom1, Triangle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_triangle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_circle(geom2, geom1)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Rectangle):
        return rectangles_intersect(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Circle):
        return rectangle_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Rectangle):
        return rectangle_intersects_circle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, Circle):
        return circles_intersect(geom1, geom2)
    if isinstance(geom1, RTrapezoid) and isinstance(geom2, RTrapezoid):
        return rtrapezoids_intersect(geom1, geom2)
    if isinstance(geom1, RTrapezoid) and isinstance(geom2, Circle):
        return rtrapezoid_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, RTrapezoid):
        return rtrapezoid_intersects_circle(geom2, geom1)
    if isinstance(geom1, RTrapezoid) and isinstance(geom2, Rectangle):
        return rtrapezoid_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, RTrapezoid):
        return rtrapezoid_intersects_rectangle(geom2, geom1)
    # Triangle intersections
    if isinstance(geom1, Triangle) and isinstance(geom2, Triangle):
        return triangles_intersect(geom1, geom2)
    if isinstance(geom1, Triangle) and isinstance(geom2, Circle):
        return triangle_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Triangle):
        return triangle_intersects_circle(geom2, geom1)
    if isinstance(geom1, Triangle) and isinstance(geom2, Rectangle):
        return triangle_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Triangle):
        return triangle_intersects_rectangle(geom2, geom1)
    if isinstance(geom1, Triangle) and isinstance(geom2, RTrapezoid):
        return triangle_intersects_rtrapezoid(geom1, geom2)
    if isinstance(geom1, RTrapezoid) and isinstance(geom2, Triangle):
        return triangle_intersects_rtrapezoid(geom2, geom1)
    if isinstance(geom1, Triangle) and isinstance(geom2, Lobject):
        return triangle_intersects_lobject(geom1, geom2)
    if isinstance(geom1, Lobject) and isinstance(geom2, Triangle):
        return triangle_intersects_lobject(geom2, geom1)
    raise NotImplementedError(
        "Intersection not implemented for geoms " f"{geom1} and {geom2}"
    )

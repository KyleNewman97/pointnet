from scipy.spatial.transform import Rotation
from torch import Tensor, nn


class RandomRotationTransform(nn.Module):
    def forward(self, points: Tensor) -> Tensor:
        """
        Applies a random rotation to all points within a point cloud.

        Parameters
        ----------
        points:
            A single point cloud with shape `(3, num_points)`.

        Returns
        -------
        rotated_points:
            Original points randomly rotated.
        """
        # Create a random rotation matrix
        rot_matrix = Rotation.random().as_matrix()
        rot_matrix_tensor = points.new_tensor(rot_matrix)

        return rot_matrix_tensor @ points

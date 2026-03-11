import unittest
import numpy as np

from makani.utils.inference.helpers import compute_crop_indices, compute_local_crop

LATS = list(map(float, range(8)))  # [0, 1, 2, 3, 4, 5, 6, 7]
LONS = list(map(float, range(8)))


class TestComputeCropIndices(unittest.TestCase):

    def test_lon_crop(self):
        lat_idx, lon_idx = compute_crop_indices((LATS, LONS), (2, 5, 0, 7))
        np.testing.assert_array_equal(lat_idx, [2, 3, 4, 5])
        np.testing.assert_array_equal(lon_idx, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_lat_crop(self):
        lat_idx, lon_idx = compute_crop_indices((LATS, LONS), (0, 7, 2, 5))
        np.testing.assert_array_equal(lat_idx, [0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(lon_idx, [2, 3, 4, 5])

    def test_combined_crop(self):
        lat_idx, lon_idx = compute_crop_indices((LATS, LONS), (1, 4, 2, 6))
        np.testing.assert_array_equal(lat_idx, [1, 2, 3, 4])
        np.testing.assert_array_equal(lon_idx, [2, 3, 4, 5, 6])

    def test_lon_wrap(self):
        """min_lon > max_lon triggers wrapping; result is ascending index order."""
        _, lon_idx = compute_crop_indices((LATS, LONS), (0, 7, 6, 2))
        np.testing.assert_array_equal(lon_idx, [0, 1, 2, 6, 7])

    def test_lon_wrap_negative_convention(self):
        """Same wrapping logic works with -180-180 convention."""
        lons_180 = [-4, -3, -2, -1, 0., 1, 2, 3]
        _, lon_idx = compute_crop_indices((LATS, lons_180), (0, 7, 2, -2))
        np.testing.assert_array_equal(lon_idx, [0, 1, 2, 6, 7])
    
    def test_lat_swapped(self):
        """Latitude bounds are auto-sorted."""
        lat_idx, _ = compute_crop_indices((LATS, LONS), (5, 2, 0, 7))
        np.testing.assert_array_equal(lat_idx, [2, 3, 4, 5])


class TestComputeLocalCrop(unittest.TestCase):
    r"""All tests share the same crop on a 6x6 grid.

    Crop (#) at lat [1,2,3], lon [0,1,4,5]. Gap at lon 2-3 simulates wrap.

         0  1  2  3  4  5
      0  .  .  .  .  .  .
      1  #  #  .  .  #  #
      2  #  #  .  .  #  #
      3  #  #  .  .  #  #
      4  .  .  .  .  .  .
      5  .  .  .  .  .  .
    """

    def setUp(self):
        self.crop = (np.array([1, 2, 3]), np.array([0, 1, 4, 5]))

    def test_tile_covers_full_crop(self):
        r"""
        +--+--+--+--+--+--+
        |.  .  .  .  .  . |
        |#  #  .  .  #  # |
        |#  #  .  .  #  # |
        |#  #  .  .  #  # |
        |.  .  .  .  .  . |
        |.  .  .  .  .  . |
        +--+--+--+--+--+--+
        """
        offset, size = (0, 0), (6, 6)
        buf_idx, out_idx = compute_local_crop(self.crop, offset, size)
        np.testing.assert_array_equal(buf_idx[0].ravel(), [1, 2, 3])
        np.testing.assert_array_equal(buf_idx[1].ravel(), [0, 1, 4, 5])
        assert out_idx == (slice(0, 3), slice(0, 4))

    def test_tile_partial_lat(self):
        r"""
         .  .  .  .  .  .
         #  #  .  .  #  #
         #  #  .  .  #  #
           +---+--+--+-+
         # |#  .  .  # |# 
         . |.  .  .  . |. 
         . |.  .  .  . |. 
           +---+--+--+-+
        """
        offset, size = (3, 1), (3, 4)
        buf_idx, out_idx = compute_local_crop(self.crop, offset, size)
        np.testing.assert_array_equal(buf_idx[0].ravel(), [0])
        np.testing.assert_array_equal(buf_idx[1].ravel(), [0, 3])
        assert out_idx == (slice(2, 3), slice(1, 3))

    def test_tile_head_lon_only(self):
        r"""
        +--+--+--+
        |.  .  . | .  .  .
        |#  #  . | .  #  #
        |#  #  . | .  #  #
        |#  #  . | .  #  #
        |.  .  . | .  .  .
        |.  .  . | .  .  .
        +--+--+--+
        """
        offset, size = (0, 0), (6, 3)
        buf_idx, out_idx = compute_local_crop(self.crop, offset, size)
        np.testing.assert_array_equal(buf_idx[0].ravel(), [1, 2, 3])
        np.testing.assert_array_equal(buf_idx[1].ravel(), [0, 1])
        assert out_idx[1] == slice(0, 2)

    def test_tile_tail_lon_only(self):
        r"""
         .  .  .  .  .  .  
                 +--+--+--+
         #  #  . |.  #  # |
         #  #  . |.  #  # |
         #  #  . |.  #  # |
         .  .  . |.  .  . |
         .  .  . |.  .  . |
                 +--+--+--+
        """
        offset, size = (1, 3), (5, 3)
        buf_idx, out_idx = compute_local_crop(self.crop, offset, size)
        np.testing.assert_array_equal(buf_idx[0].ravel(), [0, 1, 2])
        np.testing.assert_array_equal(buf_idx[1].ravel(), [1, 2])
        assert out_idx[1] == slice(2, 4)

    def test_tile_in_gap(self):
        r"""
              +--+--+
         .  . |.  . |.  .
         #  # |.  . |#  #
         #  # |.  . |#  #
         #  # |.  . |#  #
         .  . |.  . |.  .
         .  . |.  . |.  .
              +--+--+
        """
        offset, size = (0, 2), (6, 2)
        buf_idx, out_idx = compute_local_crop(self.crop, offset, size)
        assert len(buf_idx[1].ravel()) == 0
        assert out_idx[1] == slice(0, 0)

    def test_no_overlap(self):
        r"""
         .  .  .  .  .  .
         #  #  .  .  #  #
         #  #  .  .  #  #
         #  #  .  .  #  #
        +--+--+--+--+--+--+
        |.  .  .  .  .  . |
        |.  .  .  .  .  . |
        +--+--+--+--+--+--+
        """
        offset, size = (4, 0), (2, 6)
        buf_idx, out_idx = compute_local_crop(self.crop, offset, size)
        assert len(buf_idx[0].ravel()) == 0
        assert out_idx[0] == slice(0, 0)

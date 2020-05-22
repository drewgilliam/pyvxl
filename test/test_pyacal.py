import copy
import json
import numpy as np
import pickle
import unittest
import utils

from vxl import vgl
from vxl import vnl
from vxl import vpgl
from vxl.contrib import acal


def json_serializer(obj):
  try:
    return str(obj)
  except err:
    raise TypeError("Type {} not serializable".format(type(obj))) from err


class AcalBase(utils.VxlBase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.default_data = dict()

  def _set_data_full(self):
    data = copy.deepcopy(self.default_data)
    data = utils.update_nested_dict(data, getattr(self, 'set_data', {}))
    return data

  def _cls_instance(self, *args, **kwargs):
    # override to use different class creation method
    return self.cls(*args, **kwargs)

  def test_create(self):
    instance = self._cls_instance()
    self.assertIsInstance(instance, self.cls)
    self.assertAttributes(instance, self.default_data)

  @utils.skipUnlessAttr('set_data')
  def test_init(self):
    init_data = self._set_data_full()
    instance = self._cls_instance(**init_data)
    self.assertIsInstance(instance, self.cls)
    self.assertAttributes(instance, init_data)

  @utils.skipUnlessAttr('set_data')
  def test_set(self):
    instance = self._cls_instance()
    instance.set(**self.set_data)
    new_data = self._set_data_full()
    self.assertAttributes(instance, new_data)

  def test_equal(self):
    init_data = self._set_data_full()
    instance_A = self._cls_instance(**init_data)
    instance_B = self._cls_instance(**init_data)
    self.assertEqual(instance_A, instance_B)

  @utils.skipUnlessClassAttr('__getstate__')
  def test_pickle(self):
    init_data = self._set_data_full()
    insstance_A = self._cls_instance(**init_data)
    insstance_B = pickle.loads(pickle.dumps(insstance_A))
    self.assertEqual(insstance_A, insstance_B)


class acal_f_params(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.f_params
    self.default_data = {
        'epi_dist_mul': 2.5,
        'max_epi_dist': 5.0,
        'F_similar_abcd_tol': 0.01,
        'F_similar_e_tol': 1.0,
        'ray_uncertainty_tol': 50.0,
        'min_num_matches': 5,
    }
    self.set_data = {
        'epi_dist_mul': 3.0,
        'max_epi_dist': 5.5,
        'F_similar_abcd_tol': 0.51,
        'F_similar_e_tol': 1.5,
        'ray_uncertainty_tol': 50.5,
        'min_num_matches': 6,
    }


class acal_corr(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.corr
    self.default_data = {
        'id': np.uint(-1),
        'pt': vgl.point_2d(-1, -1)
    }
    self.set_data = {
        'id': 10,
        'pt': vgl.point_2d(10.0, 20.0)
    }


class acal_match_pair(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_pair
    self.default_data = {
        'corr1': {'id': np.uint(-1), 'pt': vgl.point_2d(-1, -1)},
        'corr2': {'id': np.uint(-1), 'pt': vgl.point_2d(-1, -1)},
    }
    self.set_data = {
        'corr1': {'id': 10, 'pt': vgl.point_2d(10.0, 20.0)},
        'corr2': {'id': 15, 'pt': vgl.point_2d(15.0, 25.0)},
    }


class acal_match_params(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_params
    self.default_data = {
        'min_n_tracks': 3,
        'min_n_cams': 3,
        'max_proj_error': 1.0,
        'max_uncal_proj_error': 20.0,
    }
    self.set_data = {
        'min_n_tracks': 5,
        'min_n_cams': 10,
        'max_proj_error': 10.0,
        'max_uncal_proj_error': 30.0,
    }


class acal_match_node(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_node
    self.default_data = {
      'cam_id': np.uint(-1),
      'node_depth': 0,
      'children': [],
      'self_to_child_matches': [],
    }

  @unittest.skip("not yet implemented")
  def test_equal(self):
    pass


class acal_match_tree(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_tree
    self.default_data = {
      'min_n_tracks': 1,
    }

  def _cls_instance(self, *args, **kwargs):
    root = acal.match_node()
    return acal.match_tree(root, *args, **kwargs)

  @unittest.skip("not yet implemented")
  def test_equal(self):
    pass


class acal_match_vertex(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_vertex
    self.default_data = {
      'cam_id': np.uint(-1),
      'mark': False,
    }

  @unittest.skip("not yet implemented")
  def test_equal(self):
    pass


class acal_match_edge(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_edge
    self.default_data = {
      'id': np.uint(-1),
      'matches': [],
    }

  @unittest.skip("not yet implemented")
  def test_equal(self):
    pass


class acal_match_graph(AcalBase, unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cls = acal.match_graph

  @unittest.skip("not yet implemented")
  def test_equal(self):
    pass


  @staticmethod
  def _construct_example():
    '''
    Example graph inputs for two connected components
       component "A" = 4 images/cameras (index 0-3) with 4 correspondences
       component "B" = 3 images/cameras (index 4-6) with 3 correspondences
    For each image/camera, we project 3d points into the 2d image space
    to serve as feature correspondences.
    '''

    # helper: affine camera with more defaults
    def make_affine_camera(
        rayx, rayy, rayz,   # 3D ray direction
        upx = 0.0, upy = 0.0, upz = 1.0,  # 3D up direction
        ptx = 0.0, pty = 0.0, ptz = 0.0,  # 3D stare point
        u0 = 0, v0 = 0,  # stare point image projection
        su = 1.0, sv = 1.0,  # scaling
      ):
      return vpgl.affine_camera(vgl.vector_3d(rayx, rayy, rayz),
                                vgl.vector_3d(upx, upy, upz),
                                vgl.point_3d(ptx, pty, ptz),
                                u0, v0, su, sv)

    # helper: incidence matrix with points & pairs of cams
    def make_incidence(pts, pairs, cams, cam_offset = 0):

      incidence_list = []
      for i, j in pairs:
        vec = [acal.match_pair(acal.corr(k, cams[i].project(pt)),
                               acal.corr(k, cams[j].project(pt)))
               for k, pt in enumerate(pts)]
        incidence_list.append((i + cam_offset, j + cam_offset, vec))

      return incidence_list

    # component A
    rays = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]
    camsA = [make_affine_camera(*r) for r in rays]

    pts = [(1,0,0), (0,1,0), (0,0,1), (0,0,-1)]
    ptsA = [vgl.point_3d(*p) for p in pts]

    pairsA = [(0,1),(1,2),(2,3),(3,0)]

    cam_offset = 0
    incidenceA = make_incidence(ptsA, pairsA, camsA)

    # component B
    rays = [(1,1,0), (-1,1,0), (-1,-1,0)]
    camsB = [make_affine_camera(*r) for r in rays]

    pts = [(2,0,0), (0,2,0), (0,0,2)]
    ptsB = [vgl.point_3d(*p) for p in pts]

    pairsB = [(0,1),(1,2),(2,0)]

    cam_offset += len(camsA)
    incidenceB = make_incidence(ptsB, pairsB, camsB, cam_offset)

    # total system
    pts = ptsA + ptsB
    cams = {i: c for i, c in enumerate(camsA + camsB)}
    image_paths = {i: 'image{}.tif' for i in range(len(cams))}

    incidence_matrix = incidenceA + incidenceB
    incidence_matrix = {item[0]: {item[1]: item[2]} for item in incidence_matrix}

    return (cams, image_paths, incidence_matrix)


  def test_run(self):

    cams, image_paths, incidence_matrix = self._construct_example()
    # print(json.dumps(incidence_matrix, indent = 2, default = json_serializer))

    match_graph = self._cls_instance()
    match_graph.acams = cams
    match_graph.image_paths = image_paths
    success = match_graph.load_incidence_matrix(incidence_matrix)
    self.assertTrue(success, "incidence matrix failed to load")

    match_graph.find_connected_components()
    components = match_graph.connected_components
    self.assertEqual(len(components), 2,
                     "incorrect number of connected components")
    self.assertEqual(len(components[0]), 4,
                     "incorrect size of connected component[0]")
    self.assertEqual(len(components[1]), 3,
                     "incorrect size of connected component[1]")

    match_graph.compute_focus_tracks()
    tracks = match_graph.focus_tracks
    self.assertEqual(len(tracks[0][0]), 4,
                     "incorrect size of focus_track[0][0]")
    self.assertEqual(len(tracks[1][4]), 3,
                     "incorrect size of focus_track[1][4]")

    match_graph.compute_match_trees()
    match_graph.validate_match_trees_and_set_metric()


if __name__ == '__main__':
  unittest.main()

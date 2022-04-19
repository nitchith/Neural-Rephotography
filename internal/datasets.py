# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
from os import path
import queue
import threading
import cv2
import jax
import numpy as np
from PIL import Image
from internal import utils
import pdb

def get_dataset(split, train_dir, config):
  return dataset_dict[config.dataset_loader](split, train_dir, config)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, data_dir, config):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.split = split
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    if split == 'train':
      self._train_init(config)
    elif split == 'test':
      self._test_init(config)
    else:
      raise ValueError(
          'the split argument should be either \'train\' or \'test\', set'
          'to {} here.'.format(split))
    self.batch_size = config.batch_size // jax.host_count()
    self.batching = config.batching
    self.render_path = config.render_path
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'pixels' and 'rays'.
    """
    x = self.queue.get()
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'pixels' and 'rays'.
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == 'train':
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_rays()

    if config.batching == 'all_images':
      # flatten the ray and image dimension together.
      self.images = self.images.reshape([-1, 3])
      self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                       self.rays)
    elif config.batching == 'single_image':
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = utils.namedtuple_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
    else:
      raise NotImplementedError(
          f'{config.batching} batching strategy is not implemented.')

  def _test_init(self, config):
    self._load_renderings(config)
    self._generate_rays()
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""

    if self.batching == 'all_images':
      ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
    elif self.batching == 'single_image':
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                        self.rays)
    else:
      raise NotImplementedError(
          f'{self.batching} batching strategy is not implemented.')

    return {'pixels': batch_pixels, 'rays': batch_rays}

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      return {'rays': utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
    else:
      return {
          'pixels': self.images[idx],
          'rays': utils.namedtuple_map(lambda r: r[idx], self.rays)
      }

  # TODO(bydeng): Swap this function with a more flexible camera model.
  def _generate_rays(self):
    """Generating rays for all images."""
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - self.w * 0.5 + 0.5) / self.focal,
         -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
        axis=-1)
    directions = ((camera_dirs[None, ..., None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[..., :1])
    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)

class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')
    with utils.open_file(
        path.join(self.data_dir, 'transforms_{}.json'.format(self.split)),
        'r') as fp:
      meta = json.load(fp)
    images = []
    cams = []
    for i in range(len(meta['frames'])):
      frame = meta['frames'][i]
      fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
      with utils.open_file(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if config.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif config.factor > 0:
          raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                           'set.'.format(config.factor))
      cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
      images.append(image)
    self.images = np.stack(images, axis=0)
    if config.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    camera_angle_x = float(meta['camera_angle_x'])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]



class FABlender(Dataset):
  """FA Stack Blender Dataset."""
  

  def _load_renderings(self, config):
    """Load images from disk."""

    if config.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')
    with utils.open_file(
        path.join(self.data_dir, 'transforms_{}.json'.format(self.split)),
        'r') as fp:
      meta = json.load(fp)

    images = []
    focal_dists = []
    sensor_dists = []
    apertures = []
    
    self.focal_length = 0.05 # in meters (50mm) # TODO: Inside json
    self.sensor_size = 0.036 # TODO: Inside json

    for i in range(len(meta['frames'])):
      frame = meta['frames'][i]
      fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
      with utils.open_file(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
      
      images.append(image)
     
      # Get aperture 
      fstop = frame['fstop']
      aperture = self.focal_length / (2 * fstop) # in meters, in radius
      apertures.append(aperture)

      # Get focal and sensor distance (u-focal, v-sensor)
      focal_dist = frame['focus']
      sensor_dist = (focal_dist * self.focal_length) / (focal_dist - self.focal_length) # in meters
      
      w,h,d = image.shape

      x = np.arange(w)
      y = np.arange(h)

      # Convert to range [-1, 1]
      x = (x - w * 0.5 + 0.5) / (w/2) 
      y = (y - h * 0.5 + 0.5) / (h/2)

      x *= (self.sensor_size / 2) # Scale it to meters (world )
      y *= (self.sensor_size / 2)

      xy_grid = np.stack(tuple(reversed(np.meshgrid(y,x))), axis=-1).reshape((w*h, 2))

      xy_norm = np.linalg.norm(xy_grid, ord=2, axis=1)

      xy_theta = np.arctan(xy_norm/sensor_dist)

      xy_focal_dist = focal_dist*np.cos(xy_theta)

      # theta = tan-1(norm((x,y))/sensor_dist)
      # focal_dist_pixel = focal_dist * cos (theta)
      focal_dists.append(xy_focal_dist)
      sensor_dists.append(sensor_dist)

    self.images = np.stack(images, axis=0)
    self.apertures = np.stack(apertures, axis=0)
    self.sensor_dists = np.stack(sensor_dists, axis=0)
    self.focal_dists = np.expand_dims(np.stack(focal_dists, axis=0), axis=2)

    # Shapes
    # self.images  (220, 1024, 1024, 4)
    # self.apertures  (220,)
    # self.sensor_dists  (220,)
    # self.focal_dists  (220, 1048576, 1)

    # TODO: Save near and far distances in json file
    #if self.near is None:
    self.near = self.focal_dists.min() - 0.5# 1

    #if self.far is None:
    self.far = self.focal_dists.max() + 0.5# 4

    if config.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]

    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w

    self.n_examples = self.images.shape[0]

  def _generate_rays(self):
    """Generating rays for all images."""
    x, y = np.meshgrid( 
        np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')

    self.directions = []
    self.origins = []
    self.radii = []

    for i, sensor_dist in enumerate(self.sensor_dists):

        camera_dirs = np.stack(
            [(self.sensor_size/2) * (x - self.w * 0.5 + 0.5) / (self.w * 0.5), 
            (self.sensor_size/2) * (y - self.h * 0.5 + 0.5) / (self.h * 0.5) , -sensor_dist * np.ones_like(x)],
            axis=-1)

        directions = camera_dirs

        self.directions.append(directions)

        origins = np.broadcast_to(np.zeros((3)), directions.shape)
        self.origins.append(origins)

        self.radii.append(self.apertures[i]/self.focal_dists[i])
        #print("dirs", directions.shape)
        #print("origins", origins.shape)
        #print("radii", self.radii[-1].shape)
       

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        # dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))

        # dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.


    self.origins = np.stack(self.origins, axis=0)
    self.directions = np.stack(self.directions, axis=0)
    self.radii = np.stack(self.radii, axis=0).reshape((-1, self.h, self.w, 1))
    self.viewdirs = self.directions / np.linalg.norm(self.directions, axis=-1, keepdims=True)

    # Shapes
    # self.origins  (220, 1024, 1024, 3)
    # self.directions  (220, 1024, 1024, 3)
    # self.radii  (220, 1024, 1024, 1)

    ones = np.ones_like(self.origins[..., :1])
    self.rays = utils.Rays(
        origins=self.origins,
        directions=self.viewdirs,
        viewdirs=self.viewdirs,
        radii=self.radii,
        focaldist=self.focal_dists,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)


dataset_dict = {
    'blender': Blender,
    'fablender': FABlender
}

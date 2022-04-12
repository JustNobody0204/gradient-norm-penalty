# Copyright 2020 The Authors.
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

"""Build FLAX models for image classification."""

from typing import Optional, Tuple
import flax
import jax
from jax import numpy as jnp
from jax import random

from gnp.models import pyramidnet
from gnp.models import wide_resnet
from gnp.models import wide_resnet_shakeshake
from gnp.models import wrn


_AVAILABLE_MODEL_NAMES = [
	'WideResnet28x10',
	'WideResnet28x6_ShakeShake',
	'Pyramid_ShakeDrop',
]

def init_image_model(
	prng_key: jnp.ndarray, batch_size: int, image_size: int,
	module: flax.linen.Module,
	num_channels: int = 3):

	dummy_input = jnp.zeros(shape = (batch_size, image_size, image_size, num_channels))
	variables = module.init(
	prng_key, dummy_input, train = False
	)
	state, params = variables.pop("params")
	
	return params, state


def get_model(
	model_name: str,
	batch_size: int,
	image_size: int,
	num_classes: int,
	num_channels: int = 3,
	prng_key: Optional[jnp.ndarray] = None,
	):

	if model_name == 'WideResnet28x10':
		model = wrn.WideResNet_28_10(num_outputs=num_classes)
	else:
		raise ValueError('Unrecognized model name.')
	if not prng_key:
		prng_key = random.PRNGKey(0)

	params, init_state = init_image_model(prng_key, batch_size, image_size, model, num_channels)

	return model, params, init_state

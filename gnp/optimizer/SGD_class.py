import flax
import jax
import jax.numpy as jnp
import numpy as np


class SGDOptimizer(flax.optim.OptimizerDef):

    @flax.struct.dataclass
    class HyperParams:
        learning_rate:np.ndarray
        beta: np.ndarray
        grad_norm_clip: np.ndarray
        weight_decay: np.ndarray
        nesterov : bool

    @flax.struct.dataclass
    class State:
        momentum: np.ndarray

    def __init__(self,
                 learning_rate = None,
                 beta = 0.9,
                 grad_norm_clip = None,
                 weight_decay = 0,
                 nesterov = False):
        hyper_params = SGDOptimizer.HyperParams(learning_rate, beta, grad_norm_clip, weight_decay, nesterov)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return SGDOptimizer.State(jnp.zeros_like(param))

    def apply_gradient(self, hyper_params, params, state, grads):
        step = state.step
        params_flat, treedef = jax.tree_flatten(params)
        states_flat = treedef.flatten_up_to(state.param_states)
        grads_flat = treedef.flatten_up_to(grads)

        if hyper_params.grad_norm_clip:
            grads_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
            grads_factor = jnp.minimum(1.0, hyper_params.grad_norm_clip / grads_l2)
            grads_flat = jax.tree_map(lambda param: grads_factor * param, grads_flat)

        out = [
            self.apply_param_gradient(step, hyper_params, param, state, grad)
            for param, state, grad in zip(params_flat, states_flat, grads_flat)
        ]

        new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
        new_params = jax.tree_unflatten(treedef, new_params_flat)
        new_param_states = jax.tree_unflatten(treedef, new_states_flat)
        new_state = flax.optim.OptimizerState(step + 1, new_param_states)
        return new_params, new_state

    
    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        del step
        assert hyper_params.learning_rate is not None, "no learning rate provided."

        if hyper_params.weight_decay != 0:
            grad += hyper_params.weight_decay * param
        
        momentum = state.momentum
        new_momentum = hyper_params.beta * momentum + grad
        if hyper_params.nesterov:
            d_p = grad + hyper_params.beta * new_momentum
        else:
            d_p = new_momentum

        new_param = param - hyper_params.learning_rate * d_p
        new_state = SGDOptimizer.State(new_momentum)
        return new_param, new_state



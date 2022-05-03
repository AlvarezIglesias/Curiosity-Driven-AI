import collections

import gym
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos=[['B'], [None], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], ['START']]):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def actionn(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()

    def action(self, act):
        #print("Act:", act)
        #print(self._decode_discrete_action)
        #if hasattr(act, "__len__"):
         #   return act
        #else:
        #    return self._decode_discrete_action[act].copy()
        #act = list(map(bool,act))
        act = np.array([act])
        act = np.ndarray.flatten(act)
        a = self._decode_discrete_action[act[0]].copy()
        #print(a)
        return a
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Sequence
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )
        
        # im gonna try using the whole grid as obvs space\
        # 0 -empty
        # 1- snake head
        # 2 -snake tail
        # 3- apples
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0, 3, shape=(self.size, self.size,), dtype=int),
                "direction": spaces.Discrete(4)
            }
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)
        self.direction = 0
        self.fruit_spawn = True
        self.body = [self._agent_location]
        self.distance = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        #self.change_to = self.direction

        
        # We have 5 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        grid =  np.zeros((self.size, self.size), dtype=int)
        grid[self._target_location[0], self._target_location[1]] = 3 #apple
        for i, block in enumerate(self.body):
            if i == 0:
                grid[block[0], block[1]] = 1 #head
            else:
                grid[block[0], block[1]] = 2 #tail
        return {"grid": grid, "direction": self.direction}
        

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self.direction = 0
        self.fruit_spawn = True
        self.body = [self._agent_location]

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        self.distance = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Ensure action is a plain int (stable-baselines may pass arrays)
        if isinstance(action, (np.ndarray,)):
            # 0-d array or array containing a scalar
            try:
                action = int(action)
            except Exception:
                action = int(action.item())

        reward = -0.01
        terminated = False
        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        if action != 4:
            # Prevent reversing direction (snake can't go backwards into itself)
            if not ((action == 0 and self.direction == 2) or  # right when going left
                    (action == 1 and self.direction == 3) or  # up when going down
                    (action == 2 and self.direction == 0) or  # left when going right
                    (action == 3 and self.direction == 1)):   # down when going up
                self.direction = action
        #pick direction
        
        direction = self._action_to_direction[self.direction]

        #update locations
        x,y = self._agent_location
    
        if x + direction[0] < 0 or x + direction[0] >= self.size or y + direction[1] < 0 or y + direction[1] >= self.size:
            terminated = True
            reward += -0.01
        else:
            #continue this part
            self._agent_location = (self._agent_location + direction)
            self.body.insert(0, self._agent_location) #add new head to body
            if np.array_equal(self._agent_location, self._target_location):
                reward = 1
                self.fruit_spawn = False
            else:
                self.body.pop() #remove tail
        
        if not self.fruit_spawn:
            body_positions = set(tuple(pos) for pos in self.body)
            
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
            while tuple(self._target_location) in body_positions:
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
            self.fruit_spawn = True

        
        for block in self.body[1:]: #check if head collides with body
            if np.array_equal(self._agent_location, block):
                terminated = True
                reward += -0.01

        #check if agent got closer to target
        tmp = self.distance
        self.distance = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        if self.distance < tmp:
            reward += 0.01
        else:
            reward += -0.01


        
        # # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        for body in self.body[1:]:
            pygame.draw.circle(
                canvas,
                (0, 200, 0),
                (body + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

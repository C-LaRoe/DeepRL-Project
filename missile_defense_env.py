import numpy as np
import gym
from gym import spaces
import pygame
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
import math
import time
import random as rd
X_DIM = 700
Y_DIM = 700

class Missile(pygame.sprite.Sprite):
    def __init__(self, type):
        super().__init__()
        
        if type == "agent":
            self.x_t = 0
            self.y_t = Y_DIM - 124
            self.theta = 45
            self.v = 3
        
        else:
            # Generate random trajectory in the environment for the target missile to take
            self.x_t = rd.randint(50, X_DIM-50)
            self.y_t = 0
            destination_x = rd.randint(50, X_DIM-50)
            if destination_x == self.x_t:
                self.theta = -90
            elif destination_x > self.x_t:
                self.theta = -(180/math.pi) * math.atan(Y_DIM/(destination_x - self.x_t))
            else:
                self.theta = -90 + (180/math.pi) * math.atan((destination_x - self.x_t)/Y_DIM)
            self.v = rd.randint(6,8)

        self.image = pygame.image.load("sprites/missile3.png")
        self.small_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]/7),int(self.image.get_size()[1]/7)))
        self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)
        self.rect = self.rotate_image.get_rect()
        # Set collision box coordinates
        self.rect.x = self.x_t
        self.rect.y = self.y_t

    def update_pos(self):
        rad_theta = (math.pi / 180) * self.theta
        delta_x = self.v * math.cos(rad_theta)
        delta_y = self.v * math.sin(rad_theta)
        self.x_t += delta_x
        self.y_t -= delta_y
        # Set collision box coordinates
        self.rect.x = self.x_t
        self.rect.y = self.y_t
        self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)

    def draw(self, screen):
        screen.blit(self.rotate_image, (self.x_t, self.y_t))


class Ground(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("sprites/grass.png")
        self.bigger_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]*2),int(self.image.get_size()[1])))
        self.rect = self.bigger_image.get_rect()

    def draw(self, screen):
        screen.blit(self.bigger_image, (0,Y_DIM - self.bigger_image.get_size()[1]))


class MissileEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM

        pygame.init()
        self.screen = pygame.display.set_mode([X_DIM, Y_DIM])
        self.background_color = (199, 250, 252)
        self.ground = Ground()

        self.agent_missile = Missile("agent")
        self.target_missile = Missile("target")
        explosion_img = pygame.image.load("sprites/explosion.png")
        self.explosion_img = pygame.transform.scale(explosion_img, (int(explosion_img.get_size()[0]/10),int(explosion_img.get_size()[1]/10)))
        pygame.font.init()

        # Action Space
        # Change velocity by value in range [-2, 2]
        # Change theta by value in range [-15, 15]
        self.action_space = spaces.Box(np.array([-2,-15]),np.array([2,15]))
        
        # State Space
        # self.observation_space = spaces.Dict(
        #                     {
        #                         "agent_x": spaces.Box(low=0, high=self.X_DIM, shape=()),
        #                         "agent_y": spaces.Box(low=0, high=self.Y_DIM, shape=()),
        #                         "agent_theta": spaces.Box(low=0, high=360, shape=()),
        #                         "agent_v": spaces.Box(low=3, high=12, shape=()),
        #                         "target_x": spaces.Box(low=0, high=self.X_DIM, shape=()),
        #                         "target_y": spaces.Box(low=0, high=self.Y_DIM, shape=()),
        #                         "target_theta": spaces.Box(low=0, high=360, shape=()),
        #                         "target_v": spaces.Box(low=3, high=12, shape=())
        #                     })
        self.observation_space = spaces.Box(low=np.array([
                                                            0,
                                                            0,
                                                            -360,
                                                            3,
                                                            0,
                                                            0,
                                                            -360,
                                                            3
                                                        ]).astype(np.float32),
                                            high=np.array([
                                                            self.X_DIM,
                                                            self.Y_DIM,
                                                            360,
                                                            15,
                                                            self.X_DIM,
                                                            self.Y_DIM,
                                                            360,
                                                            12,
                                                        ]).astype(np.float32)
                                                    )

    def step(self, action):
        # print(action)       
        # Need to update agent missile theta and v based on the action passed as a parameter
        self.agent_missile.theta += action[1] # Change theta
        if self.agent_missile.v + action[0] >= 3 and self.agent_missile.v + action[0] <= 15:
            self.agent_missile.v += action[0] # Change v

        self.agent_missile.update_pos()
        self.target_missile.update_pos()
        self.agent_missile.draw(self.screen)
        self.target_missile.draw(self.screen)
        self.ground.draw(self.screen)
        pygame.display.flip()
        # pygame.draw.rect(screen, (0,255,0) , target_missile.rect) # Draw collision box of target

        self.screen.fill(self.background_color)
        # If agent missile has hit ground or gone outside the bounds of the environment 
        if self.agent_missile.y_t > Y_DIM-124 or self.agent_missile.y_t < 0 or self.agent_missile.x_t > X_DIM or self.agent_missile.x_t < 0:
            reward = -100
            done = True

        # Target missile hit ground
        elif self.target_missile.y_t > Y_DIM-124:
            self.screen.blit(self.explosion_img, (self.target_missile.x_t-15, self.target_missile.y_t))
            pygame.display.flip()
            time.sleep(0.1)
            reward = -100
            done = True

        # If there is a missile to missile collision
        elif self.agent_missile.rect.colliderect(self.target_missile.rect):
            self.screen.blit(self.explosion_img, (self.target_missile.x_t, self.target_missile.y_t))
            pygame.display.flip()
            time.sleep(0.1)
            reward = 100
            done = True

        else: # No collisions occured
            # Give reward based on current distance between agent and target missile
            reward = 10 / math.sqrt((self.target_missile.x_t - self.agent_missile.x_t) ** 2 + (self.target_missile.y_t - self.agent_missile.y_t) ** 2)
            done = False

        time.sleep(0.009)
        pygame.display.flip()

        # observation = {
        #             "agent_x": self.agent_missile.x_t,
        #             "agent_y": self.agent_missile.y_t,
        #             "agent_theta": self.agent_missile.theta,
        #             "agent_v": self.agent_missile.v,
        #             "target_x": self.target_missile.x_t,
        #             "target_y": self.target_missile.y_t,
        #             "target_theta": self.target_missile.theta,
        #             "target_v": self.target_missile.v,
        #         }
        observation = np.array([self.agent_missile.x_t,self.agent_missile.y_t,self.agent_missile.theta,self.agent_missile.v,
                                self.target_missile.x_t,self.target_missile.y_t,self.target_missile.theta,self.target_missile.v], dtype=np.float32)
        return observation, reward, done, {}

    def reset(self):
        self.agent_missile = Missile(type="agent")
        self.target_missile = Missile(type="target")
        observation = np.array([self.agent_missile.x_t, self.agent_missile.y_t, self.agent_missile.theta, self.agent_missile.v,
                                self.target_missile.x_t, self.target_missile.y_t, self.target_missile.theta, self.target_missile.v], dtype=np.float32)

        # print(self.observation_space.shape)
        # print(self.observation_space.dtype)
        # print(observation)
        return observation
        # return {
        #             "agent_x": self.agent_missile.x_t,
        #             "agent_y": self.agent_missile.y_t,
        #             "agent_theta": self.agent_missile.theta,
        #             "agent_v": self.agent_missile.v,
        #             "target_x": self.target_missile.x_t,
        #             "target_y": self.target_missile.y_t,
        #             "target_theta": self.target_missile.theta,
        #             "target_v": self.target_missile.v,
        #         }   


def main():
    env = MissileEnv()
    # check_env(env)
    t1 = time.time()
    # model = A2C.load("a2c_missile")
    model = A2C("MlpPolicy", env)
    model.learn(total_timesteps=10000)
    # model = A2C("MlpPolicy", env, verbose=1)
    model.save("a2c_missile")
    t2 = time.time()
    print(t2 - t1, "seconds")

    print("DONE TRAINING")

    # del model # remove to demonstrate saving and loading

    # model = PPO.load("a2c_missile")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()

if __name__ == "__main__":
    main()

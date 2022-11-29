import numpy as np
import gym
from gym import spaces
import pygame
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import math
import time
import random as rd
X_DIM = 800
Y_DIM = 800
# Hyperparameters
k_a = -0.2
k_z = -1
k_vr = -2
r_d = 20
k_r = 10

class Missile(pygame.sprite.Sprite):
    def __init__(self, type):
        super().__init__()
        
        if type == "agent":
            self.x_t = 100
            self.y_t = Y_DIM - 124
            self.theta = 45
            self.v = 5
        
        else:
            # Generate random trajectory in the environment for the target missile to take
            # self.x_t = rd.randint(50, X_DIM-50)
            self.x_t = 400
            self.y_t = 200
            # self.y_t = 0
            destination_x = rd.randint(50, X_DIM-50)
            # destination_x = 360
            if destination_x == self.x_t:
                self.theta = -90
            elif destination_x > self.x_t:
                self.theta = -(180/math.pi) * math.atan(Y_DIM/(destination_x - self.x_t))
            else:
                self.theta = -90 + (180/math.pi) * math.atan((destination_x - self.x_t)/Y_DIM)
            if self.theta < 0:
                self.theta += 360 
            self.theta = 280
            # self.v = rd.randint(3,4)
            self.v = 0

        self.image = pygame.image.load("sprites/missile3.png")
        self.small_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]/7),int(self.image.get_size()[1]/7)))
        self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)
        self.rect = self.rotate_image.get_rect()
        # Set collision box coordinates
        self.rect.x = self.x_t
        self.rect.y = self.y_t

    def update_pos(self):
        if self.theta < 0:
            self.theta += 360
        elif self.theta > 360:
            self.theta = self.theta % 360
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
    metadata = {'render.modes': ['human']}
    def __init__(self, display=True):
        super().__init__()

        self.display = display
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.z_init = 0

        if display:
            pygame.init()
            self.screen = pygame.display.set_mode([X_DIM, Y_DIM])
            self.background_color = (199, 250, 252)
            # self.ground = Ground()

        self.t = 1
        self.agent_missile = Missile("agent")
        self.target_missile = Missile("target")
        explosion_img = pygame.image.load("sprites/explosion.png")
        self.explosion_img = pygame.transform.scale(explosion_img, (int(explosion_img.get_size()[0]/10),int(explosion_img.get_size()[1]/10)))
        pygame.font.init()

        # Action Space
        self.action_space = spaces.Box(np.array([-15]),np.array([15]))
        
        # State Space
        self.observation_space = spaces.Box(low=np.array([
                                                            0,
                                                            0,
                                                            -50,
                                                            -360,
                                                        ]).astype(np.float32),
                                            high=np.array([
                                                            3000,
                                                            360,
                                                            50,
                                                            360
                                                        ]).astype(np.float32)
                                                    )

    def step(self, action):
        # print(action)       
        # Need to update agent missile theta and v based on the action passed as a parameter
        self.agent_missile.theta += action[0] # Change theta

        x_dif = self.target_missile.x_t - self.agent_missile.x_t
        y_dif = self.target_missile.y_t - self.agent_missile.y_t
        r = math.sqrt((x_dif)**2 + (y_dif)**2)
        lamb = np.arctan2(y_dif, x_dif)
        lamb_deg = (180/math.pi) * lamb
        if lamb_deg < 0:
            lamb_deg += 360
        print("r:", r)
        print("Lambda:", lamb_deg)
        agent_rad_theta = self.agent_missile.theta * (math.pi/180)
        target_rad_theta = self.target_missile.theta * (math.pi/180)
        r_dot = self.target_missile.v * math.cos(target_rad_theta - lamb) - self.agent_missile.v * math.cos(agent_rad_theta - lamb)
        if r == 0:
            lambda_dot = 0
        else:
            lambda_dot = (self.target_missile.v * math.sin(target_rad_theta - lamb) - self.agent_missile.v * math.sin(agent_rad_theta - lamb)) / (r + 0.000001)

        self.agent_missile.update_pos()
        # self.target_missile.update_pos()
        if self.display:
            self.agent_missile.draw(self.screen)
            self.target_missile.draw(self.screen)
            # self.ground.draw(self.screen)
            pygame.display.flip()
            # pygame.draw.rect(screen, (0,255,0) , target_missile.rect) # Draw collision box of target
            self.screen.fill(self.background_color)

        # Target missile hit ground
        if self.target_missile.y_t > Y_DIM-124:
            if self.display:
                self.screen.blit(self.explosion_img, (self.target_missile.x_t-15, self.target_missile.y_t))
                pygame.display.flip()
                time.sleep(0.1)
            reward = -1000
            done = True

        # If there is a missile to missile collision
        elif self.agent_missile.rect.colliderect(self.target_missile.rect):
            if self.display:
                self.screen.blit(self.explosion_img, (self.target_missile.x_t, self.target_missile.y_t))
                pygame.display.flip()
                time.sleep(0.1)
            reward = 1000
            done = True

        else: # No collisions occured
            # Give reward based on current distance between agent and target missile
            # reward = r_dot
            z = ((r**2) * lambda_dot) / math.sqrt((r_dot**2) + (r**2) * (lambda_dot**2))
            reward_z = k_z * ((z / self.z_init)**2)
            if r_dot > 0:
                reward_vr = k_vr
            else:
                reward_vr = 0
            
            if r <= r_d:
                reward_r = k_r
            else:
                reward_r = 0

            reward = reward_z + reward_vr + reward_r
            self.t += 1
            done = False

        if self.t == 400:
            done = True

        print("t =", self.t, "Reward: ", round(reward,4))
        if self.display:
            time.sleep(0.009)
            pygame.display.flip()

        observation = np.array([r, lamb, r_dot, lambda_dot], dtype=np.float32)
        return observation, reward, done, {}

    def reset(self):
        self.agent_missile = Missile(type="agent")
        self.target_missile = Missile(type="target")
        self.t = 1
        x_dif = self.target_missile.x_t - self.agent_missile.x_t
        y_dif = self.target_missile.y_t - self.agent_missile.y_t
        r = math.sqrt((x_dif)**2 + (y_dif)**2)
        lamb = np.arctan2(y_dif, x_dif)
        agent_rad_theta = self.agent_missile.theta * (math.pi/180)
        target_rad_theta = self.target_missile.theta * (math.pi/180)
        r_dot = self.target_missile.v * math.cos(target_rad_theta - lamb) - self.agent_missile.v * math.cos(agent_rad_theta - lamb)
        if r == 0:
            lambda_dot = 0
        else:
            lambda_dot = (self.target_missile.v * math.sin(target_rad_theta - lamb) - self.agent_missile.v * math.sin(agent_rad_theta - lamb)) / (r + 0.000001)
        observation = np.array([r, lamb, r_dot, lambda_dot], dtype=np.float32)

        self.z_init = ((r**2) * lambda_dot) / math.sqrt((r_dot**2) + (r**2) * (lambda_dot**2))

        return observation 

    def set_display(self, show):
        if not self.display:
            if show:
                self.display = True
                pygame.init()
                self.screen = pygame.display.set_mode([X_DIM, Y_DIM])
                self.background_color = (199, 250, 252)
                # self.ground = Ground()
            else:
                return
        else:
            if show:
                return
            else:
                self.display = False
                pygame.quit()


def main():
    env = MissileEnv(display=True)
    env.reset()
    # model = A2C.load("a2c_missile")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=4 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    print("Training....")
    model.learn(total_timesteps=5000)
    env.set_display(True)
    obs = env.reset()
    time.sleep(10)
    print("Testing....")
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
    env.set_display(False)

    print("Training....")
    model.learn(total_timesteps=1000)

    env.set_display(True)
    print("Testing....")
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
    env.set_display(False)

    # model.save("a2c_missile")


if __name__ == "__main__":
    main()

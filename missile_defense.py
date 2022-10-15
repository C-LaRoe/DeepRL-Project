import pygame
import math
import time
import random as rd

X_DIM = 700
Y_DIM = 700

class Missile(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        
        self.x_t = rd.randint(50, X_DIM-50)
        self.y_t = 0
        destination_x = rd.randint(50, X_DIM-50)
        if destination_x == self.x_t:
            self.theta = -90
        elif destination_x > self.x_t:
            self.theta = -(180/math.pi) * math.atan(Y_DIM/(destination_x - self.x_t))
        else:
            self.theta = -90 + (180/math.pi) * math.atan((destination_x - self.x_t)/Y_DIM)
        self.v = rd.randint(5,10)

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

    def draw(self, screen):
        screen.blit(self.rotate_image, (self.x_t, self.y_t))


class UserMissile(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # User missile initial position should be in the bottom left corner
        self.x_t = 0
        self.y_t = Y_DIM - 124
        self.theta = 45
        self.v = 15
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

    def draw(self, screen):
        screen.blit(self.rotate_image, (self.x_t, self.y_t))

    def control(self, event):
        # Make sure to update angle of picture in this
        if event.key == pygame.K_UP:
            if self.v < 24: # 24 is max velocity
                self.v += 3
        elif event.key == pygame.K_DOWN:
            if self.v > 6: # 6 is min velocity
                self.v -= 3
        elif event.key == pygame.K_LEFT:
            self.theta += 15 
            self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)
        elif event.key == pygame.K_RIGHT:
            self.theta -= 15
            self.rotate_image = pygame.transform.rotate(self.small_image, self.theta)



class Ground(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("sprites/grass.png")
        self.bigger_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]*2),int(self.image.get_size()[1])))
        self.rect = self.bigger_image.get_rect()

    def draw(self, screen):
        screen.blit(self.bigger_image, (0,Y_DIM - self.bigger_image.get_size()[1]))

class Explosion(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("sprites/explosion.png")
        self.small_image = pygame.transform.scale(self.image, (int(self.image.get_size()[0]*3),int(self.image.get_size()[1]*3)))
        # self.rect = self.bigger_image.get_rect()


pygame.init()
screen = pygame.display.set_mode([X_DIM,Y_DIM])
background_color = (199, 250, 252)
target_missile = Missile()
user_missile = UserMissile()
ground = Ground()
explosion_img = pygame.image.load("sprites/explosion.png")
small_explosion_img = pygame.transform.scale(explosion_img, (int(explosion_img.get_size()[0]/10),int(explosion_img.get_size()[1]/10)))
pygame.font.init()
my_font = pygame.font.SysFont("Comic Sans MS", 30)
score = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            user_missile.control(event)

    screen.fill(background_color)
    ground.draw(screen)
    if user_missile.y_t > Y_DIM-124 or user_missile.y_t < 0 or user_missile.x_t > X_DIM or user_missile.x_t < 0:
        # reset user missile
        user_missile = UserMissile()

    # Check collisions
    if user_missile.rect.colliderect(target_missile.rect):
        screen.blit(small_explosion_img, (target_missile.x_t, target_missile.y_t))
        pygame.display.flip()
        score += 50
        time.sleep(1)
        target_missile = Missile()
        user_missile = UserMissile()

    score_text = my_font.render("Score: " + str(score), False, (255,255,255))
    screen.blit(score_text, (20, Y_DIM-40))

    user_missile.draw(screen)
    # pygame.draw.rect(screen, (0,255,0) , user_missile.rect) # Draw collision box of guided missile
    user_missile.update_pos()

    target_missile.draw(screen)
    # pygame.draw.rect(screen, (0,255,0) , target_missile.rect) # Draw collision box of target
    # Missile Hit Ground
    if target_missile.y_t > Y_DIM-124:
        screen.blit(small_explosion_img, (target_missile.x_t-15, target_missile.y_t))
        pygame.display.flip()
        score -= 10
        time.sleep(1)
        target_missile = Missile()
        user_missile = UserMissile()

    target_missile.update_pos()

        
    # check if target missile went off screen
    if target_missile.x_t > X_DIM or target_missile.x_t < 0:
        # reset target missile
        target_missile = Missile()

    time.sleep(0.05)
    pygame.display.flip()
    
pygame.quit()










# def get_v0(x0, y0, xf, yf, t):
#     g = 9.8
#     v_x0 = (xf - x0) / t
#     v_y0 = ((yf - y0) + 0.5 * g * (t**2)) / t 
#     return v_x0, v_y0

# x0 = 10
# y0 = 490
# t = 10

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.MOUSEBUTTONDOWN:
#             target_pos = event.pos
#             v_x0, v_y0 = get_v0(x0, y0, target_pos[0], y0, t)
#             v = math.sqrt(v_x0**2 + v_y0**2)
#             for i in np.arange(0,t+1,0.1):
#                 x_t = x0 + v_x0 * i
#                 y_t = y0 - v_y0 * i + (0.5 * 9.8 * i**2)
#                 # print("x =",x_t, "y =", y_t, "t =", i)
#                 screen.fill((255,255,255))
#                 pygame.draw.circle(screen, (0,0,255), (x0,y0), 10)
#                 pygame.draw.circle(screen, (255,0,0), (x_t,y_t), 5)
                
#                 pygame.display.flip()
#                 time.sleep(0.01)

#     screen.fill((255,255,255))
#     pygame.draw.circle(screen, (0,0,255), (x0,y0), 10)
#     pygame.display.flip()


# pygame.quit()
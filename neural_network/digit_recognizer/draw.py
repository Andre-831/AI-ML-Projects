import pygame
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mnist_cnn_model.h5")

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 280, 280
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RADIUS = 8

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a Digit (Enter=Predict, Space=Clear)")
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(WHITE)

font = pygame.font.SysFont(None, 48)
drawing = False
predicted_digit = None


def preprocess_image(surface):
    # Convert surface to grayscale 28x28 image, normalized and reshaped
    raw_str = pygame.image.tostring(surface, 'RGB')
    img = np.frombuffer(raw_str, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = 255 - img  # invert colors: white bg to black bg

    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    countours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if countours:
        x, y, w, h = cv2.boundingRect(max(countours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    else:
        digit = img 
    
    #resize to fit in 20x20 box then pad it to 28x28
    h,w = img.shape
    if h > w:
        new_h = 20
        new_w = int(20 * w / h)
    else:
        new_w = 20
        new_h = int(20 * h / w)
    
    digit = cv2.resize(img, (new_w, new_h))

    #pad the image to 28x28
    padded = np.pad(digit,
                (((28 - new_h) // 2, (28 - new_h + 1) // 2),  # top, bottom
                 ((28 - new_w) // 2, (28 - new_w + 1) // 2)), # left, right
                mode='constant', constant_values=0)

    
    padded = padded / 255.0  # normalize to [0, 1]
    return padded.reshape(1, 28, 28, 1)  # reshape for model input

    

def predict_digit(surface):
    # Convert surface to grayscale 28x28 image, normalized and reshaped
    img = preprocess_image(surface)

    # Predict digit and confidence
    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    if confidence > 0.75:
        return np.argmax(prediction)
    else:
        return None

while True:
    screen.fill(WHITE)
    screen.blit(canvas, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                canvas.fill(WHITE)
                predicted_digit = None

            elif event.key == pygame.K_RETURN:
                predicted_digit = predict_digit(canvas)

    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(canvas, BLACK, mouse_pos, RADIUS)

    if predicted_digit is not None:
        text = font.render(f"Predicted: {predicted_digit}", True, (0, 128, 0))
        screen.blit(text, (10, 10))

    pygame.display.flip()

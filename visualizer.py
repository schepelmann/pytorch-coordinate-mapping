# Example file showing a basic pygame "game loop"
import argparse

import numpy as np
import pygame
import torch

from functions import decrease, increase, reverse
from network import NeuralNetwork
from settings import SCREEN_SIZE
from utils import normalize


def visualize(model_path: str, func: callable) -> None:
    """
    Visualizes the model prediction with a red dot and the ground truth value
    with green dot.
    """
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()
    running = True

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        mouse_pos = np.array(pygame.mouse.get_pos())

        # normalize mouse data for the neural network
        mouse_x, mouse_y = normalize(mouse_pos[0], mouse_pos[1])
        
        model = NeuralNetwork()
        model.load_state_dict(torch.load(model_path))
        x, y = model(torch.tensor((mouse_x, mouse_y), dtype=torch.float32))

        predicted_pos = (
            int(x.item() * SCREEN_SIZE[0]),
            int(y.item() * SCREEN_SIZE[1]),
        )
        actual_pos = func(mouse_pos, np.array(screen.get_size()))
        
        # RENDER YOUR GAME HERE
        pygame.draw.circle(screen, 'green', actual_pos, 4)
        pygame.draw.circle(screen, 'red', predicted_pos, 4)

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='./saved_model.pt',
        help='Path to the saved model.'
    )

    parser.add_argument(
        '--function', '-f',
        type=str,
        choices=['increase', 'decrease', 'reverse'],
        default='increase',
        help='Choose the function that should be visualized as ground truth.'
    )

    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    if opt.function == 'increase':
        visualize(opt.model_path, increase)
    if opt.function == 'decrease':
        visualize(opt.model_path, decrease)
    if opt.function == 'reverse':
        visualize(opt.model_path, reverse)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

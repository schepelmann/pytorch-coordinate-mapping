import argparse
import csv
import os

import numpy as np
import pygame

from functions import decrease, increase, reverse


def collect_data(output_path: str, func: callable) -> None:
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    with open(os.path.join(output_path, 'data.csv'), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        while running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # fill the screen with a color to wipe away anything from last frame
            screen.fill("white")

            mouse_pos = np.array(pygame.mouse.get_pos())

            new_mouse_pos = func(mouse_pos, np.array(screen.get_size()))

            csv_writer.writerow(
                [mouse_pos[0], mouse_pos[1], new_mouse_pos[0], new_mouse_pos[1]]
            )

            pygame.draw.circle(screen, 'green', new_mouse_pos, 8)

            pygame.display.flip()

            clock.tick(60)  # limits FPS to 60

    pygame.quit()


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./',
        help='Directory to store the data in.'
    )

    parser.add_argument(
        '--function', '-f',
        type=str,
        choices=['increase', 'decrease', 'reverse'],
        default='increase',
        help='Choose the function that should be learned.'
    )

    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    if opt.function == 'increase':
        collect_data(opt.output_dir, increase)
    if opt.function == 'decrease':
        collect_data(opt.output_dir, decrease)
    if opt.function == 'reverse':
        collect_data(opt.output_dir, reverse)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
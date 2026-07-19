# from tests.run_gui import *
# main()

from trainer.trainer import Trainer
from gui.draw_app import DrawApp


def main():

    trainer = Trainer()

    app = DrawApp()
    app.run()


if __name__ == "__main__":
    main()
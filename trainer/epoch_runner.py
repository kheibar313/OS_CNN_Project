# """
# Epoch Runner.

# Responsible for executing exactly one training epoch.

# This module isolates TensorFlow's model.fit() call
# from the Trainer and allows EpochThread to execute
# a single epoch independently.
# """

# from __future__ import annotations

# from tensorflow.keras import Model
# from tensorflow.keras.callbacks import History


# class EpochRunner:
#     """
#     Executes a single training epoch.
#     """

#     def __init__(self, model: Model) -> None:

#         self.model = model

#     def run_epoch(
#         self,
#         x_train,
#         y_train,
#         x_test,
#         y_test,
#         batch_size: int,
#     ) -> History:
#         """
#         Execute one epoch of training.
#         """

#         history = self.model.fit(
#             x=x_train,
#             y=y_train,
#             epochs=1,
#             batch_size=batch_size,
#             validation_data=(
#                 x_test,
#                 y_test,
#             ),
#             verbose=1,
#         )

#         return history

"""
Epoch Runner.

Responsible for executing exactly one training epoch.
"""

from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History


class EpochRunner:
    """
    Executes exactly one training epoch.
    """

    def __init__(
        self,
        model: Model,
    ) -> None:

        self.model = model

    def run_epoch(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size: int,
    ) -> tuple[History, float, float]:
        """
        Execute one epoch.

        Returns
        -------
        (
            history,
            loss,
            accuracy,
        )
        """

        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(
                x_test,
                y_test,
            ),
            verbose=1,
        )

        loss = history.history["loss"][-1]

        accuracy = history.history["accuracy"][-1]

        return (
            history,
            float(loss),
            float(accuracy),
        )
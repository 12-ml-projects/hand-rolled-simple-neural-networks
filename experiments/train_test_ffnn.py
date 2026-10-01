import tempfile
from pathlib import Path
from typing import TypeAlias

import click
import mlflow
import numpy as np

from src.custom_types import Array
from src.ffnn import mnist
from src.ffnn.criterion import CrossEntropyLoss
from src.ffnn.ffnn import Mlp
from src.ffnn.optimiser import SGD
from src.ffnn.utils import train
from src.tensor import Tensor

EXPERIMENT = "ffnn-mnist"

Scores: TypeAlias = tuple[float, float]


@click.command()
@click.option("--epochs", default=10, help="Number of passes over the training set")
@click.option("--batch-size", default=64, help="Samples per gradient step")
@click.option("--hidden-size", default=128, help="Width of the hidden layer")
@click.option("--learning-rate", default=0.1, help="SGD step size")
@click.option("--seed", default=0, help="Seeds initialisation and shuffling")
def main(
    epochs: int,
    batch_size: int,
    hidden_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    (train_features, train_labels), (test_features, test_labels) = mnist.load()
    train_batches = mnist.batches(train_features, train_labels, batch_size, rng)

    model = Mlp(train_features.shape[1], hidden_size, mnist.CLASSES, rng)
    optimiser = SGD(model.parameters(), learning_rate)
    criterion = CrossEntropyLoss(mnist.CLASSES)

    mlflow.set_experiment(EXPERIMENT)

    run_name = f"h{hidden_size}-b{batch_size}-lr{learning_rate}-s{seed}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "parameters": sum(p.value.size for p in model.parameters()),
                "train_samples": len(train_labels),
                "test_samples": len(test_labels),
            }
        )

        losses = train(model, train_batches, criterion, optimiser, epochs)

        for epoch, running_loss in enumerate(losses, start=1):
            train_loss, train_accuracy = _evaluate(
                model, criterion, train_features, train_labels
            )
            test_loss, test_accuracy = _evaluate(
                model, criterion, test_features, test_labels
            )

            mlflow.log_metrics(
                {
                    "train_loss_running": running_loss,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "generalisation_gap": train_accuracy - test_accuracy,
                },
                step=epoch,
            )

            click.echo(
                f"epoch {epoch:3d}  "
                f"train {train_loss:.4f}/{train_accuracy:.4f}  "
                f"test {test_loss:.4f}/{test_accuracy:.4f}  "
                f"gap {train_accuracy - test_accuracy:.4f}"
            )

        _log_weights(model)


def _evaluate(
    model: Mlp, criterion: CrossEntropyLoss, features: Array, labels: Array
) -> Scores:
    logits = model(Tensor(features, requires_grad=False))
    predictions = np.argmax(logits.value, axis=-1)

    return (
        float(criterion(logits, labels).value),
        float(np.mean(predictions == labels)),
    )


def _log_weights(model: Mlp) -> None:
    """Keep the trained weights, in parameters() order.

    Otherwise a run records what happened and throws away what it produced.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.npz"
        np.savez(path, *[parameter.value for parameter in model.parameters()])

        mlflow.log_artifact(str(path))


if __name__ == "__main__":
    main()

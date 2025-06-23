"""Implement the .train function."""

from transformers import Trainer, TrainingArguments


def train(model, loss_fn, train_dataset, eval_dataset,
          optimizer='adam',
          epochs=120, lr=0.1, weight_decay=5e-4
          ):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    args = TrainingArguments(
        "training",
        optim=optimizer,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=epochs,
        weight_decay=weight_decay
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=loss_fn.metric,
    )

    trainer.train()

    # Get the training stats
    return trainer.evaluate()

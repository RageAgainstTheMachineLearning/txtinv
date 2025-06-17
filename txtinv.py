import time
import torch
import numpy as np

# ------ Local imports ------ #
from data import load_data, num_classes
from model import get_bert_model, get_bert_tokenizer, tokenize_and_align_labels, get_loss_fn
from options import options
from plot import display_batch, display_reconstruction
from reconstructor import GradientReconstructor
from utils import system_startup


def extract_ground_truth(dataset, embedding_matrix, batch_size, setup):
    """
    Extracts a ground truth sample from the dataset. This is going
    to be the target sample for the reconstruction.

    Args:

        dataset (Dataset): The dataset to extract the ground truth from.
        embedding_matrix (torch.Tensor): The embedding matrix of the model.
        batch_size (int): The batch size for the model.
        setup (dict): The setup dictionary containing device and dtype.
    """
    if batch_size == 1:
        target_id = np.random.randint(len(dataset))
        gt = dataset[target_id]
        gt_data = embedding_matrix[gt["input_ids"]].unsqueeze(0).to(**setup)
        gt_ids = torch.as_tensor(
            gt["input_ids"], dtype=torch.long).unsqueeze(0).to(**setup)
        gt_label = torch.as_tensor(
            gt["labels"], dtype=torch.long).unsqueeze(0).to(**setup)
        gt_mask = torch.as_tensor(
            gt["attention_mask"], dtype=torch.long).unsqueeze(0).to(**setup)
    else:
        gt_data, gt_label, gt_mask = [], [], []
        target_id = np.random.randint(len(dataset))
        while len(gt_label) < batch_size:
            gt = dataset[target_id]
            target_id += 1
            gt_data.append(
                embedding_matrix[gt["input_ids"]].unsqueeze(0).to(**setup))
            gt_ids.append(torch.as_tensor(
                gt["input_ids"], dtype=torch.long).to(**setup))
            gt_label.append(torch.as_tensor(
                gt["labels"], dtype=torch.long).to(**setup))
            gt_mask.append(torch.as_tensor(
                gt["attention_mask"], dtype=torch.long).to(**setup))
        gt_data = torch.stack(gt_data)
        gt_ids = torch.stack(gt_ids)
        gt_label = torch.stack(gt_label)
        gt_mask = torch.stack(gt_mask)
    return gt_ids, gt_data, gt_label, gt_mask


def extract_gradient(
        model,
        loss_fn,
        gt_data,
        gt_label,
        gt_mask,
        num_labels,
):
    """
    Extracts the gradient of the loss with respect to the model
    parameters using the ground truth data.

    Args:
        model (nn.Module): The model to extract the gradient from.
        loss_fn (nn.Module): The loss function to use.
        gt_data (torch.Tensor): The ground truth data.
        gt_label (torch.Tensor): The ground truth labels.
        gt_mask (torch.Tensor): The ground truth mask.
        num_labels (int): The number of labels in the dataset.
    """

    model.train()
    model.zero_grad()

    inputs = {
        "inputs_embeds": gt_data,
        "attention_mask": gt_mask,
    }

    # Forward pass.
    outputs = model(**inputs)

    # Compute logits.
    logits = outputs.logits
    logits = logits.view(-1, num_labels)

    # Compute loss.
    loss = loss_fn(logits, gt_label.view(-1))

    # Compute gradients.
    gt_gradient = torch.autograd.grad(
        loss, model.parameters(), create_graph=True)
    gt_gradient = [grad.detach() for grad in gt_gradient]
    gt_gradnorm = torch.stack([g.norm() for g in gt_gradient]).mean()

    print(f"Gradient Norm: {gt_gradnorm}")

    return gt_gradient


if __name__ == "__main__":
    args = options().parse_args()

    setup = system_startup()
    start_time = time.time()

    dataset = load_data(args.dataset)
    loss_fn = get_loss_fn()

    # print some data
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")

    num_labels = num_classes(args.dataset)

    model = get_bert_model(args.model, num_labels)
    model.to(**setup)
    model.eval()

    embedding_matrix = model.get_input_embeddings().weight
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    # TODO Sanity check: run the model.

    tokenizer = get_bert_tokenizer("bert-base-cased")
    train = dataset["train"].map(
        lambda batch: tokenize_and_align_labels(tokenizer, batch), batched=True
    )

    # Extracting original data and labels.
    gt_ids, gt_data, gt_label, gt_mask = extract_ground_truth(
        train, embedding_matrix, args.batch_size, setup)

    print(f"Ground truth data shape: {gt_data.shape}")
    print(f"Ground truth label shape: {gt_label.shape}")
    print(f"Ground truth mask shape: {gt_mask.shape}")

    display_batch(gt_ids, tokenizer)

    gt_gradient = extract_gradient(
        model,
        loss_fn,
        gt_data,
        gt_label,
        gt_mask,
        num_labels
    )

    gt_label = gt_label if not args.reconstruct_label else None
    gt_mask = gt_mask if not args.reconstruct_label else None

    reconstructor = GradientReconstructor(model, num_labels=num_labels, tokenizer=tokenizer,
                                          loss_fn=loss_fn, embedding_matrix=embedding_matrix,
                                          embedding_dim=hidden_size, batch_size=args.batch_size,
                                          optimizer=args.optimizer,
                                          max_iter=args.max_iter, lr_decay=args.lr_decay,
                                          cost_fn=args.cost_fn,
                                          idlg=args.idlg)

    data, stats = reconstructor.reconstruct(
        gt_gradient, gt_label, gt_mask, trials=args.trials)

    display_batch(gt_ids, tokenizer)
    display_reconstruction(data, tokenizer, embedding_matrix)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")

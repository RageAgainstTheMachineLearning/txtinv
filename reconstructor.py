from collections import defaultdict
import time
import torch

from plot import display_reconstruction


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, num_labels, tokenizer, embedding_dim, embedding_matrix,
                 batch_size=1, optimizer='adam', max_iter=100,
                 lr_decay=False, cost_fn='sim',
                 loss_fn=torch.nn.CrossEntropyLoss(reduction='mean'), idlg=False):
        """Initialize with algorithm setup."""
        self.model = model
        self.num_labels = num_labels

        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix

        self.tokenizer = tokenizer

        self.setup = dict(device=next(model.parameters()).device,
                          dtype=next(model.parameters()).dtype)

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.cost_fn = cost_fn
        self.loss_fn = loss_fn

        if idlg:
            raise NotImplementedError("iDLG trick not implemented")

    def reconstruct(self, gt_gradient, gt_label=None, mask=None, trials=1):
        """Reconstruct data from gradient."""
        start_time = time.time()
        stats = defaultdict(list)

        dim = None if gt_label is None else gt_label.shape[1]

        x = self._init_data(trials, dim, mask)
        scores = torch.zeros(trials)

        try:
            for trial in range(trials):
                x_trial, y_trial = self._run_trial(
                    x[trial], gt_label, gt_gradient, mask)
                x[trial] = x_trial

                scores[trial] = self._score_trial(
                    x_trial, y_trial, gt_gradient, mask)
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        print('Choosing optimal result ...')
        # guard against NaN/-Inf scores?
        scores = scores[torch.isfinite(scores)]
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        stats['score'] = scores[optimal_index].item()
        x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_data(self, trials, dim=512, mask=None):
        """
        Initializes the data for the reconstruction process.
        Args:
            trials (int): Number of trials to run.
            dim (int): Dimension of the input.
            mask (torch.Tensor): Mask for the input data. It has shape (batch_size, dim).
        Returns:
            A tensor of shape (trials, batch_size, dim).
        """
        dim = (dim, self.embedding_dim)
        data = torch.randn((trials, self.batch_size, *dim), **self.setup)
        if mask is not None:
            # Let's get the embedding of the padding token.
            pad_token = self.tokenizer.pad_token_id
            pad_token_embedding = self.embedding_matrix[pad_token]

            # The mask is a tensor of shape (batch_size, dim). When it's
            # zero, we want to set the corresponding embedding to the
            # padding token embedding, otherwise we want to keep it to
            # what's in the data.
            mask = mask.unsqueeze(-1).expand_as(data)
            data = torch.where(mask == 0, pad_token_embedding, data)
        return data

    def _run_trial(self, x_trial, y_trial, gt_gradient, mask=None):
        x_trial = x_trial.clone().detach().requires_grad_(True)

        if y_trial is None:
            inputs = {
                'inputs_embeds': x_trial,
                'attention_mask': mask
            }

            out = self.model(**inputs).logits

            y_trial = torch.randn(out.shape[1]).to(
                **self.setup).requires_grad_(True)

            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam([x_trial, y_trial], lr=0.1)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    [x_trial, y_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.optimizer == 'LBFGS':
                optimizer = torch.optim.LBFGS(
                    [x_trial, y_trial], lr=1, line_search_fn='strong_wolfe')
            else:
                raise ValueError()
        else:
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=0.1)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    [x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.optimizer == 'LBFGS':
                optimizer = torch.optim.LBFGS(
                    [x_trial], lr=1, line_search_fn='strong_wolfe')
            else:
                raise ValueError()

        max_iter = self.max_iter

        if self.lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iter // 2.667, max_iter // 1.6,
                                                                         max_iter // 1.142], gamma=0.1)
        try:
            for iteration in range(max_iter):
                closure = self._gradient_closure(
                    optimizer, x_trial, y_trial, gt_gradient, mask)
                rec_loss = optimizer.step(closure)
                if self.lr_decay:
                    scheduler.step()

                with torch.no_grad():
                    # Project into the embedding space (between -1 and 1).
                    x_trial.data = torch.clamp(x_trial.data, -1, 1)

                    if (iteration + 1 == max_iter) or iteration % 10 == 0:
                        print(
                            f'It: {iteration}. Rec. loss: {rec_loss.item():2.8f}.')
                    if iteration % 100 == 0:
                        display_reconstruction(
                            x_trial, self.tokenizer, self.embedding_matrix)
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        return x_trial.detach(), y_trial

    def _gradient_closure(self, optimizer, x_trial, y_trial, gt_gradient, mask=None):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            inputs = {
                "inputs_embeds": x_trial,
                "attention_mask": mask
            }

            # Compute logits.
            logits = self.model(**inputs).logits
            logits = logits.view(-1, self.num_labels)

            # Compute loss.
            loss = self.loss_fn(logits, y_trial.view(-1))

            # Compute gradients. TODO: Check the allow_unused.
            gradient = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True, allow_unused=True, materialize_grads=True)

            # Compute reconstruction loss.
            rec_loss = self._reconstruction_costs(
                gradient, gt_gradient)
            rec_loss.backward()

            return rec_loss
        return closure

    def _score_trial(self, x_trial, y_trial, gt_gradient, mask=None):
        self.model.zero_grad()
        x_trial.grad = None

        inputs = {
            'inputs_embeds': x_trial,
            'attention_mask': mask
        }

        # Compute logits.
        logits = self.model(**inputs).logits
        logits = logits.view(-1, self.num_labels)

        loss = self.loss_fn(logits, y_trial.view(-1))

        gradient = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=False, allow_unused=True, materialize_grads=True)

        return self._reconstruction_costs(gradient, gt_gradient)

    def _reconstruction_costs(self, grad, gt_grad):
        """Input gradient is given data."""

        indices = torch.arange(len(gt_grad))  # Default indices.
        weights = gt_grad[0].new_ones(len(gt_grad))  # Same weight.

        pnorm = [0, 0]
        costs = 0
        for i in indices:
            if self.cost_fn == 'l2':
                costs += ((grad[i] - gt_grad[i]).pow(2)).sum() * weights[i]
            elif self.cost_fn == 'l1':
                costs += ((grad[i] - gt_grad[i]).abs()).sum() * weights[i]
            elif self.cost_fn == 'max':
                costs += ((grad[i] - gt_grad[i]).abs()).max() * weights[i]
            elif self.cost_fn == 'sim':
                costs -= (grad[i] * gt_grad[i]).sum() * weights[i]
                pnorm[0] += grad[i].pow(2).sum() * weights[i]
                pnorm[1] += gt_grad[i].pow(2).sum() * weights[i]
            elif self.cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(grad[i].flatten(),
                                                                   gt_grad[i].flatten(
                ),
                    0, 1e-10) * weights[i]
        if self.cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        return costs

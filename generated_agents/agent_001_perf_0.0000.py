class MetaLearningAgent(BaseAgent):
    def __init__(self, model, meta_lr=1e-3, inner_lr=1e-2, num_inner_steps=5, meta_batch_size=4):
        super().__init__(model)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_batch_size = meta_batch_size
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.inner_optimizer = optim.SGD(self.model.parameters(), lr=inner_lr)
        self.meta_losses = deque(maxlen=100)

    def forward(self, task):
        # Split the task into meta-train and meta-test sets
        train_inputs, train_outputs, test_inputs, test_outputs = task.split_for_meta_learning(self.meta_batch_size)

        # Perform inner loop updates
        for _ in range(self.num_inner_steps):
            self.inner_optimizer.zero_grad()
            train_preds = self.model(train_inputs)
            inner_loss = self.model.loss_function(train_preds, train_outputs)
            inner_loss.backward()
            self.inner_optimizer.step()

        # Compute meta-loss on the meta-test set
        test_preds = self.model(test_inputs)
        meta_loss = self.model.loss_function(test_preds, test_outputs)
        self.meta_losses.append(meta_loss.item())

        # Update the meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return test_preds

    def update_meta_lr(self, new_lr):
        self.meta_lr = new_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)

    def update_inner_lr(self, new_lr):
        self.inner_lr = new_lr
        self.inner_optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
```

This `MetaLearningAgent` class implements a meta-learning algorithm that can adapt quickly to new tasks by leveraging prior experience. It inherits from the `BaseAgent` class and includes the following key components:

- `__init__` method: Initializes the meta-learning hyperparameters, including the meta learning rate, inner learning rate, number of inner steps, and meta batch size. It also creates the meta optimizer and inner optimizer for updating the model parameters.

- `forward` method: Implements the forward pass of the meta-learning algorithm. It splits the task into meta-train and meta-test sets, performs inner loop updates on the meta-train set for a specified number of steps, computes the meta-loss on the meta-test set, and updates the meta-parameters using the meta-optimizer.

- `update_meta_lr` and `update_inner_lr` methods: Allow for dynamic adjustment of the meta learning rate and inner learning rate during training.

The meta-learning algorithm works by simulating the process of adapting to a new task within the inner loop updates, and then updating the model's meta-parameters based on the performance on the meta-test set. This approach enables the agent to quickly adapt to new tasks by leveraging prior knowledge gained from previous tasks.
class HyperNetworkAgent(BaseAgent):
    def __init__(self, model, meta_lr=1e-3, inner_lr=1e-2, num_inner_steps=5, meta_batch_size=4, hypernet_hidden_size=64):
        super().__init__(model)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_batch_size = meta_batch_size
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.inner_optimizer = optim.SGD(self.model.parameters(), lr=inner_lr)
        self.meta_losses = deque(maxlen=100)

        # Initialize the hypernetwork
        self.hypernet = HyperNetwork(model.num_params, hypernet_hidden_size)
        self.hypernet_optimizer = optim.Adam(self.hypernet.parameters(), lr=meta_lr)

    def forward(self, task):
        # Split the task into meta-train and meta-test sets
        train_inputs, train_outputs, test_inputs, test_outputs = task.split_for_meta_learning(self.meta_batch_size)

        # Generate task-specific weights using the hypernetwork
        task_weights = self.hypernet(task.task_embedding)
        self.model.update_weights(task_weights)

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

        # Update the meta-parameters and hypernetwork
        self.meta_optimizer.zero_grad()
        self.hypernet_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        self.hypernet_optimizer.step()

        return test_preds

    def update_meta_lr(self, new_lr):
        self.meta_lr = new_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.hypernet_optimizer = optim.Adam(self.hypernet.parameters(), lr=self.meta_lr)

    def update_inner_lr(self, new_lr):
        self.inner_lr = new_lr
        self.inner_optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)

class HyperNetwork(nn.Module):
    def __init__(self, num_params, hidden_size):
        super().__init__()
        self.num_params = num_params
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_params)
        self.relu = nn.ReLU()

    def forward(self, task_embedding):
        x = self.fc1(task_embedding)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1)
```

This `HyperNetworkAgent` class extends the `MetaLearningAgent` by incorporating a hypernetwork to generate task-specific weights for the main model. The key components are:

- `__init__` method: In addition to the meta-learning hyperparameters, it initializes a `HyperNetwork` module and a separate optimizer for updating the hypernetwork
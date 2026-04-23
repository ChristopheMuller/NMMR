import numpy as np
import torch
import torch.optim as optim

from src.models.NMMR.NMMR_model import MLP_for_NMMR
from src.models.NMMR.NMMR_loss import NMMR_loss
from src.models.NMMR.kernel_utils import calculate_kernel_matrix


class NMMRCATEEstimator:
    """
    A scikit-learn style wrapper for the NMMR model to easily plug into 
    your benchmarking suite.
    """

    def __init__(self, train_params: dict, random_seed: int = 42):
        """
        Args:
            train_params: Dictionary containing hyperparameters. 
                          Requires: 'network_width', 'network_depth'.
                          Optional defaults: 'n_epochs', 'batch_size', 'learning_rate', 
                                             'l2_penalty', 'loss_name'
            random_seed:  Seed for reproducibility.
        """
        self.train_params = train_params
        self.n_epochs = train_params.get('n_epochs', 100)
        self.batch_size = train_params.get('batch_size', 256)
        self.learning_rate = train_params.get('learning_rate', 1e-3)
        self.l2_penalty = train_params.get('l2_penalty', 1e-4)
        self.loss_name = train_params.get('loss_name', 'U_statistic')
        self.random_seed = random_seed

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None

        self.W_train = None
        self.dim_A = 1

    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray, W: np.ndarray, Z: np.ndarray):
        """
        Fits the NMMR model.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # 1. Convert everything to Torch tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        A_t = torch.tensor(A, dtype=torch.float32, device=self.device).view(-1, 1)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device).view(-1, 1)
        W_t = torch.tensor(W, dtype=torch.float32, device=self.device)
        Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)

        # Save W_train to marginalize over during inference
        self.W_train = W_t

        self.dim_A = A_t.shape[1]
        dim_W = W_t.shape[1]
        dim_X = X_t.shape[1]
        
        # 2. Instantiate the MLP architecture
        input_size = self.dim_A + dim_W + dim_X
        self.model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_penalty
        )

        n_samples = X_t.shape[0]

        # 3. Training Loop
        self.model.train()
        for epoch in range(self.n_epochs):
            permutation = torch.randperm(n_samples, device=self.device)

            for i in range(0, n_samples, self.batch_size):
                indices = permutation[i:i + self.batch_size]

                batch_X = X_t[indices]
                batch_A = A_t[indices]
                batch_W = W_t[indices]
                batch_Z = Z_t[indices]
                batch_Y = Y_t[indices]

                optimizer.zero_grad()

                # Model Forward Pass (A, W, X)
                batch_inputs = torch.cat((batch_A, batch_W, batch_X), dim=1)
                pred_Y = self.model(batch_inputs)

                # Construct Kernel matrix using (A, Z, X)
                kernel_inputs_train = torch.cat((batch_A, batch_Z, batch_X), dim=1)
                kernel_matrix_train = calculate_kernel_matrix(kernel_inputs_train)

                # NMMR U-Statistic or V-Statistic loss
                loss = NMMR_loss(pred_Y, batch_Y, kernel_matrix_train, self.loss_name)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray, W=None, Z=None) -> np.ndarray:
        """
        Predicts the CATE for given covariates X.
        W and Z are ignored at predict time since we marginalize over W_train.
        """
        if self.model is None or self.W_train is None:
            raise RuntimeError("You must call fit() before predict().")

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_test = X_t.shape[0]
        num_W = self.W_train.shape[0]

        cate_preds = np.zeros(n_test)

        self.model.eval()
        with torch.no_grad():
            batch_size = 128  # chunking to prevent OOM errors on large sets
            for i in range(0, n_test, batch_size):
                X_batch = X_t[i:i + batch_size]
                b_size = X_batch.shape[0]

                # To compute E[Y(1) - Y(0) | X], we marginalize over the empirical distribution of W.
                # Expand X and W so every test point is paired with every W from the training set.
                X_expanded = X_batch.repeat_interleave(num_W, dim=0)
                W_expanded = self.W_train.repeat(b_size, 1)

                A1 = torch.ones((b_size * num_W, self.dim_A), device=self.device)
                A0 = torch.zeros((b_size * num_W, self.dim_A), device=self.device)

                in1 = torch.cat((A1, W_expanded, X_expanded), dim=1)
                in0 = torch.cat((A0, W_expanded, X_expanded), dim=1)

                # Forward pass both counterfactuals
                y1 = self.model(in1).view(b_size, num_W)
                y0 = self.model(in0).view(b_size, num_W)

                # Average the difference over the W dimension
                cate_batch = (y1 - y0).mean(dim=1).cpu().numpy()
                cate_preds[i:i + batch_size] = cate_batch

        return cate_preds
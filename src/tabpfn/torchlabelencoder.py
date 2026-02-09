import torch
class TorchLabelEncoder:
    def fit(self, y):
        self.classes_ = sorted(set(y))
        self.class_to_index_ = {
            c: i for i, c in enumerate(self.classes_)
        }
        return self

    def transform(self, y):
        return torch.tensor(
            [self.class_to_index_[v] for v in y],
            dtype=torch.long
        )

    def inverse_transform(self, y):
        return [self.classes_[i] for i in y.tolist()]

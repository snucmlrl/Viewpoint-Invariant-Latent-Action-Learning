## Prepare Datasets for UniSkill

We provide the `BaseDataset` class in [base_dataset.py](./base_dataset.py) as a skeleton structure for datasets.

### Structure of BaseDataset
```python
class BaseDataset:
    def __init__(self):
        # Set the data path for your dataset.
        # Define the minimum and maximum prediction horizon (range of k).

    def _prepare_data(self):
        # Implement the data loading logic for your dataset.
        # This may vary depending on your data structure.

    def __len__(self):
        # Return the dataset length.

    def __getitem__(self, index):
        # Implement the logic to retrieve a single data sample.

    def read_images(self):
        # Implement the method to read images from videos.
```
To create a custom dataset, inherit from `BaseDataset` and implement the necessary methods.
You can refer to other dataset implementations under [diffusion/dataset](../dataset/) for guidance.
import openml
import pandas as pd

mnist_dataset = openml.datasets.get_dataset(554)

X, y, _, _ = mnist_dataset.get_data(target=mnist_dataset.default_target_attribute)

mnist_df = pd.DataFrame(X)
mnist_df['label'] = y

mnist_df.to_csv('../../data/mnist/mnist.csv', index=False)
print("mnist.csv downloaded.")

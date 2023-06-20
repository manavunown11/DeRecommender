import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))
embedding_dimension = 32
user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    return self.task(user_embeddings, positive_movie_embeddings)

  model = MovielensModel(user_model, movie_model)
  model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
  cached_train = train.shuffle(100_000).batch(8192).cache()
  cached_test = test.batch(4096).cache()
  model.fit(cached_train, epochs=3)

  model.evaluate(cached_test, return_dict=True)

  index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
  index.index_from_dataset(
      tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
  )

  _, titles = index(tf.constant(["46"]))
  print(f"Recommendations for user 46: {titles[0, :3]}")

  with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "model")
      tf.saved_model.save(index, path)
      loaded = tf.saved_model.load(path)
      scores, titles = loaded(["42"])
      print(f"Recommendations: {titles[0][:3]}")

scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

_, titles = scann_index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")


with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  tf.saved_model.save(
      index,
      path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )
  loaded = tf.saved_model.load(path)

  scores, titles = loaded(["42"])

  print(f"Recommendations: {titles[0][:3]}")
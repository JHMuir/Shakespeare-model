predicted_id = tensorflow.random.categorical(tensorflow.math.log(tensorflow.expand_dims(predictions, 0)), num_samples=1)[0, 0].numpy()

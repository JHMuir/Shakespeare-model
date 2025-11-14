predicted_id = tensorflow.random.categorical(tensorflow.math.log(tensorflow.expand_dims(predictions, 0)), num_samples=1)[0, 0].numpy()


```
def generate_text(model, phrase, length, temperature):
    seq_length = model.input_shape[1]
    input_phrase = [char_to_index[c] for c in phrase]
    input_phrase = tensorflow.expand_dims(input_phrase, 0)
    # input_phrase = tensorflow.convert_to_tensor(input_phrase, )
    print(input_phrase)
    
    generated = []
    
    for i in range(length):
        if input_phrase.shape[1] > seq_length:
            input_phrase = input_phrase[:, -seq_length:]
        elif input_phrase.shape[1] < seq_length:
            pad_len = seq_length - input_phrase.shape[1]
            input_phrase = tensorflow.pad(input_phrase, [[0, 0], [pad_len, 0]])
            
        predictions = model(input_phrase)
        predictions = tensorflow.squeeze(predictions, 0)
        
        predictions = predictions / temperature
        predicted_id = tensorflow.random.categorical(tensorflow.math.log(tensorflow.expand_dims(predictions, 0)), num_samples=1)[0, 0].numpy()
        
        input_phrase = tensorflow.concat([input_phrase, [[predicted_id]]], axis=1)
        
        generated.append(index_to_char[predicted_id])
        
    return phrase + "".join(generated)
    
print(generate_text(model, "Where art thou?!", 200, 1))
```

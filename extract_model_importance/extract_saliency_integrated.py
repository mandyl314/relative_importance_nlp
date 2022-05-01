


import numpy as np
import tensorflow as tf
import scipy.special


# The code for calculating sensitivity is based on the integrated gradients method
def compute_sensitivity(model, embedding_matrix, tokenizer, text, nsamples = 25):
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    vocab_size = embedding_matrix.get_shape()[0]
    sensitivity_data = []

    # Iteratively mask each token in the input
    for masked_token_index in range(len(token_ids)):

        if masked_token_index == 0:
            sensitivity_data.append({'token': '[CLR]', 'sensitivity': [1] + [0] * (len(token_ids) - 1)})

        elif masked_token_index == len(token_ids) - 1:
            sensitivity_data.append({ 'token': '[SEP]', 'sensitivity': [0] * (len(token_ids) - 1) + [1]})

        # Get the actual token
        else:
            target_token = tokenizer.convert_ids_to_tokens(token_ids[masked_token_index])
            # integers are not differentable, so use a one-hot encoding of the intput
            token_ids_tensor = tf.constant([token_ids[0:masked_token_index] + [tokenizer.mask_token_id] + token_ids[masked_token_index + 1:]], dtype='int32')
            token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)
            
            # To select the correct output, create a masking tensor.
            # tf.gather_nd could also be used, but this is easier.
            output_mask = np.zeros((1, len(token_ids), vocab_size))
            output_mask[0, masked_token_index, token_ids[masked_token_index]] = 1
            output_mask_tensor = tf.constant(output_mask, dtype='float32')
            
            #number of steps for integrated gradients
          
            # Compute gradient of the logits of the correct target, w.r.t. the input
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(token_ids_tensor_one_hot)
                inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix)
                baseline = tf.zeros_like(inputs_embeds)
                alphas = tf.linspace(start = 0.0, stop = 1.0, num = nsamples + 1)
                alphas_x = alphas[:, tf.newaxis, tf.newaxis]
                baseline_x = tf.expand_dims(baseline, axis = 0)
                input_x = tf.expand_dims(inputs_embeds, axis = 0)
                delta = input_x - baseline_x
                interpolated_inputs = baseline_x + alphas_x * delta
                predict = model({"inputs_embeds": tf.squeeze(interpolated_inputs, axis = 0) }).logits
                predict_mask_correct_token = tf.reduce_sum(predict * output_mask_tensor)

            # compute the sensitivity and take l2 norm
                
            sensitivity_non_normalized = tape.gradient(predict_mask_correct_token, token_ids_tensor_one_hot)
            sensitivity_non_normalized = tf.reduce_mean(sensitivity_non_normalized, axis = 0, keepdims = True)
            sensitivity_non_normalized = tf.norm(sensitivity_non_normalized, axis = 2)

            # Normalize by the max
            sensitivity_tensor = (sensitivity_non_normalized / tf.reduce_max(sensitivity_non_normalized))
            sensitivity = sensitivity_tensor[0].numpy().tolist()

            sensitivity_data.append({'token': target_token,'sensitivity': sensitivity })

    return sensitivity_data

# We calculate relative saliency by summing the sensitivity a token has with all other tokens
def extract_relative_saliency(model, embeddings,tokenizer, sentence):
    sensitivity_data = compute_sensitivity(model, embeddings, tokenizer, sentence)

    distributed_sensitivity = np.asarray([entry["sensitivity"] for entry in sensitivity_data])
    tokens = [entry["token"] for entry in sensitivity_data]

    # For each token, I sum the sensitivity values it has with all other tokens
    saliency = np.sum(distributed_sensitivity, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    # saliency = scipy.special.softmax(saliency)
    return tokens, saliency



# eval.py

import tensorflow as tf

if __name__ == '__main__':
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.h5')
    
    # Load evaluation dataset
    eval_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'data/eval',
        image_size=(64, 64),
        batch_size=2,
        label_mode='binary'
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(eval_ds)
    print("Evaluation Accuracy:", accuracy)

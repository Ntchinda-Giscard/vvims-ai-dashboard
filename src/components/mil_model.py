import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


class AttentionMILPoolingWithConstraints(layers.Layer):
    """
    Enhanced Attention-based MIL pooling layer with sparsity and smoothness constraints.
    """

    def __init__(self, L=512, D=128, dropout=0.0, sparsity_regularizer=0.001, smoothness_regularizer=0.001, **kwargs):
        super(AttentionMILPoolingWithConstraints, self).__init__(**kwargs)
        self.L = L  # Hidden layer size
        self.D = D  # Attention embedding dimension
        self.dropout = dropout
        self.sparsity_regularizer = sparsity_regularizer
        self.smoothness_regularizer = smoothness_regularizer

        # Attention mechanism layers
        self.attention_V = layers.Dense(self.L, activation='tanh')
        self.attention_U = layers.Dense(self.D, activation='sigmoid')
        self.attention_weights = layers.Dense(1, activation=None, use_bias=False,
                                              kernel_regularizer=regularizers.l1(self.sparsity_regularizer))
        self.dropout_layer = layers.Dropout(self.dropout)

    def call(self, inputs, training=None):
        # inputs shape: (batch_size, instances, features)

        # Apply attention mechanism
        v = self.attention_V(inputs)  # (batch_size, instances, L)
        v = self.dropout_layer(v, training=training)
        u = self.attention_U(v)  # (batch_size, instances, D)
        u = self.dropout_layer(u, training=training)

        # Calculate attention scores
        a = self.attention_weights(u)  # (batch_size, instances, 1)

        # Apply softmax to get attention weights
        a = tf.nn.softmax(a, axis=1)  # Normalize over instances

        # Add smoothness constraint as an activity regularizer
        if self.smoothness_regularizer > 0:
            # Calculate differences between adjacent attention weights
            a_shifted = tf.roll(a, shift=1, axis=1)
            smoothness_loss = tf.reduce_sum(tf.abs(a - a_shifted), axis=1)
            self.add_loss(self.smoothness_regularizer * tf.reduce_mean(smoothness_loss))

        # Compute weighted mean of instances using attention weights
        z = tf.reduce_sum(inputs * a, axis=1)  # (batch_size, features)

        return z, a

    def get_config(self):
        config = super().get_config()
        config.update({
            'L': self.L,
            'D': self.D,
            'dropout': self.dropout,
            'sparsity_regularizer': self.sparsity_regularizer,
            'smoothness_regularizer': self.smoothness_regularizer
        })
        return config


def mil_ranking_loss(y_true, y_pred, margin=1.0):
    """
    MIL ranking loss function.
    Ensures positive bags have higher scores than negative bags.

    Args:
        y_true: True labels (1 for positive bags, 0 for negative bags)
        y_pred: Predicted scores
        margin: Margin between positive and negative scores
    """
    # Reshape to ensure correct dimensions
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Find positive and negative indices
    pos_indices = tf.where(tf.equal(y_true, 1.0))
    neg_indices = tf.where(tf.equal(y_true, 0.0))

    # Get positive and negative predictions
    pos_preds = tf.gather(y_pred, pos_indices)
    neg_preds = tf.gather(y_pred, neg_indices)

    # Compute all pairwise differences
    pos_preds_expanded = tf.expand_dims(pos_preds, 1)
    neg_preds_expanded = tf.expand_dims(neg_preds, 0)

    # Calculate loss: max(0, margin - (pos_pred - neg_pred))
    differences = margin - (pos_preds_expanded - neg_preds_expanded)
    zeros = tf.zeros_like(differences)
    loss = tf.reduce_mean(tf.maximum(zeros, differences))

    return loss


def create_enhanced_mil_model(input_shape, num_classes=1, use_ranking_loss=True, use_feature_extractor=True):
    """
    Create an enhanced MIL model with attention pooling and constraints.

    Args:
        input_shape: Tuple of (instances, features)
        num_classes: Number of output classes
        use_ranking_loss: Whether to use ranking loss instead of BCE
        use_feature_extractor: Whether to use additional feature extraction layers

    Returns:
        A TensorFlow model for MIL with enhanced capabilities
    """
    # Define input layer
    inputs = layers.Input(shape=input_shape)

    # Feature extraction
    if use_feature_extractor:
        x = layers.TimeDistributed(layers.Dense(512, activation='relu'))(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    else:
        x = inputs

    # Apply enhanced attention-based MIL pooling with constraints
    x, attention_weights = AttentionMILPoolingWithConstraints(
        sparsity_regularizer=0.001,
        smoothness_regularizer=0.001
    )(x)

    # Bag-level classification
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile with appropriate loss
    if use_ranking_loss:
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=mil_ranking_loss,
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    else:
        loss_func = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss_func,
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

    # Create a second model that outputs attention weights for analysis
    attention_model = Model(inputs=model.input, outputs=attention_weights)

    return model, attention_model


def prepare_video_data_for_mil(video_features, labels, bag_size=16, stride=8):
    """
    Convert video features into MIL bags with temporal segments.
    Handles variable length videos with sliding window approach.

    Args:
        video_features: List of video feature arrays, each of shape (num_frames, features)
        labels: List of video labels (0 or 1)
        bag_size: Number of consecutive frames to include in each bag
        stride: Number of frames to shift when creating overlapping bags

    Returns:
        bags: List of bags, each bag is an array of shape (bag_size, features)
        bag_labels: Array of bag labels (same as video label)
    """
    bags = []
    bag_labels = []
    video_indices = []  # To keep track of which video each bag comes from

    for video_idx, (features, label) in enumerate(zip(video_features, labels)):
        # Handle variable length videos
        num_frames = len(features)

        # Create bags using sliding window
        for start_idx in range(0, num_frames - bag_size + 1, stride):
            end_idx = start_idx + bag_size
            bag = features[start_idx:end_idx]
            bags.append(bag)
            bag_labels.append(label)
            video_indices.append(video_idx)

    # Convert to numpy arrays
    bag_labels = np.array(bag_labels)

    # If all bags have the same shape, convert to one big numpy array
    if all(bag.shape == bags[0].shape for bag in bags):
        bags = np.array(bags)

    return bags, bag_labels, video_indices


def train_model_with_video_data(video_features, video_labels, bag_size=16, stride=8, epochs=10, batch_size=32):
    """
    Train an enhanced MIL model with video data.

    Args:
        video_features: List of video feature arrays
        video_labels: List of video labels
        bag_size: Number of consecutive frames in each bag
        stride: Stride for sliding window
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Trained model and attention model
    """
    # Prepare data
    bags, bag_labels, video_indices = prepare_video_data_for_mil(
        video_features, video_labels, bag_size, stride
    )

    # Split into train and validation sets (by video to avoid data leakage)
    unique_videos = np.unique(video_indices)
    np.random.shuffle(unique_videos)
    split_idx = int(0.8 * len(unique_videos))
    train_videos = unique_videos[:split_idx]
    val_videos = unique_videos[split_idx:]

    train_mask = np.isin(video_indices, train_videos)
    val_mask = np.isin(video_indices, val_videos)

    train_bags = bags[train_mask] if isinstance(bags, np.ndarray) else [bags[i] for i, m in enumerate(train_mask) if m]
    train_labels = bag_labels[train_mask]

    val_bags = bags[val_mask] if isinstance(bags, np.ndarray) else [bags[i] for i, m in enumerate(val_mask) if m]
    val_labels = bag_labels[val_mask]

    # Pad variable-length bags if necessary
    if not isinstance(bags, np.ndarray):
        # Find max length
        max_len = max(max(len(bag) for bag in train_bags), max(len(bag) for bag in val_bags))
        feature_dim = train_bags[0].shape[1]

        # Pad train bags
        padded_train_bags = np.zeros((len(train_bags), max_len, feature_dim))
        for i, bag in enumerate(train_bags):
            padded_train_bags[i, :len(bag), :] = bag

        # Pad validation bags
        padded_val_bags = np.zeros((len(val_bags), max_len, feature_dim))
        for i, bag in enumerate(val_bags):
            padded_val_bags[i, :len(bag), :] = bag

        train_bags = padded_train_bags
        val_bags = padded_val_bags

    # Create and train model
    input_shape = train_bags.shape[1:]
    model, attention_model = create_enhanced_mil_model(
        input_shape,
        use_ranking_loss=True,
        use_feature_extractor=True
    )

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    # Train model
    model.fit(
        train_bags, train_labels,
        validation_data=(val_bags, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return model, attention_model


# Example usage for video data with synthetic features
def example_video_mil_pipeline():
    # Generate synthetic video data (simulate C3D features)
    num_videos = 100
    frames_per_video = np.random.randint(50, 200, size=num_videos)  # Variable length videos
    feature_dim = 512  # Typical C3D feature dimension

    # Create random video features and labels
    video_features = [np.random.randn(frames, feature_dim) for frames in frames_per_video]
    video_labels = np.random.binomial(1, 0.3, size=num_videos)

    # Train model
    model, attention_model = train_model_with_video_data(
        video_features,
        video_labels,
        bag_size=16,
        stride=8,
        epochs=10,
        batch_size=32
    )

    # Analyze attention weights for a sample video
    sample_idx = 0
    sample_video = video_features[sample_idx]

    # Create bags from sample video
    bags = []
    for start_idx in range(0, len(sample_video) - 16 + 1, 8):
        bags.append(sample_video[start_idx:start_idx + 16])

    # Pad bags to same length if needed
    if len(bags) > 0:
        bags_array = np.array(bags)
        attention_weights = attention_model.predict(bags_array)

        # Visualize attention weights (in a real implementation, you'd plot this)
        for i, weights in enumerate(attention_weights):
            print(f"Segment {i}: Top 3 frames by attention weight:",
                  np.argsort(weights[:, 0])[-3:])

    return model, attention_model


if __name__ == "__main__":
    example_video_mil_pipeline()
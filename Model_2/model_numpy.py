from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).parent
FEATURES = PROJECT / "features"
MODEL_PATH = PROJECT / "models" / "cnn_spektrogram_numpy.npz"
CLASS_NAMES = ["good", "broken", "heavyload"]
SEED = 42
VALIDATION_FRACTION = 0.2 # 20 % til validering
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_FILTERS = 8


# Laste inn features
def load_features():
    required = ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "cond_test.npy")
    missing = [name for name in required if not (FEATURES / name).is_file()]
    if missing:
        sys.exit(
            f"Mangler feature-filer i {FEATURES}: {', '.join(missing)}\n"
            "Kjør først preprocess.py."
        )

    X_train = np.load(FEATURES / "X_train.npy").astype(np.float32)
    y_train = np.load(FEATURES / "y_train.npy").astype(np.int64)
    X_test = np.load(FEATURES / "X_test.npy").astype(np.float32)
    y_test = np.load(FEATURES / "y_test.npy").astype(np.int64)
    conditions = np.load(FEATURES / "cond_test.npy")

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("X_train og X_test må ha form (antall, høyde, bredde).")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Trenings- og testdata må ha samme spektrogramdimensjoner.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Antall spektrogrammer må stemme med antall etiketter.")
    if len(y_test) != len(conditions):
        raise ValueError("Antall testetiketter må stemme med cond_test.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature-dataene inneholder ugyldige eller uendelige verdier.")
    if not np.isin(y_train, range(len(CLASS_NAMES))).all():
        raise ValueError("y_train inneholder klasseetiketter utenfor 0, 1, 2.")
    if not np.isin(y_test, range(len(CLASS_NAMES))).all():
        raise ValueError("y_test inneholder klasseetiketter utenfor 0, 1, 2.")

    return X_train, y_train, X_test, y_test, conditions

# Deles i et treningssett og et valideringssett
def stratified_split(labels, validation_fraction):
    train_indices = []
    validation_indices = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        n_validation = max(1, int(round(len(indices) * validation_fraction)))
        if n_validation >= len(indices):
            raise ValueError("For få eksempler i en klasse til å lage valideringssett.")
        # Siden klippene er sammenhengende segementer fra en enkelt opptak
        # per klasse, så hold de siste segmente utenfor for å redusere
        # lekkasje til naboen/nærliggende datapunkter.
        train_indices.extend(indices[:-n_validation])
        validation_indices.extend(indices[-n_validation:])
    return np.asarray(train_indices), np.asarray(validation_indices)


# Fremoverberegning konvulsjonslag
def conv_forward(x, weights, bias):
    pad_h = weights.shape[2] // 2
    pad_w = weights.shape[3] // 2
    padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (weights.shape[2], weights.shape[3]), axis=(1, 2)
    )
    output = np.einsum("nhwckl,ockl->nhwo", windows, weights, optimize=True)
    output += bias
    return output, (windows, weights, x.shape, pad_h, pad_w)

# Konvolusjonlag
# Data rundt kanten,legger til padding på input-data, finner små
# vinduer med data som kan som behandles, multipliserere vinduene med
# vekt og summerer. Bias som brukes til "backpropagation"
def conv_backward(gradient, cache):
    windows, weights, input_shape, pad_h, pad_w = cache
    weight_gradient = np.einsum(
        "nhwckl,nhwo->ockl", windows, gradient, optimize=True
    )
    bias_gradient = gradient.sum(axis=(0, 1, 2))

    padded_gradient = np.zeros(
        (input_shape[0], input_shape[1] + 2 * pad_h,
         input_shape[2] + 2 * pad_w, input_shape[3]),
        dtype=gradient.dtype,
    )
    for row in range(weights.shape[2]):
        for col in range(weights.shape[3]):
            contribution = np.einsum(
                "nhwo,oc->nhwc", gradient, weights[:, :, row, col], optimize=True
            )
            padded_gradient[
                :, row:row + input_shape[1], col:col + input_shape[2], :
            ] += contribution
    input_gradient = padded_gradient[
        :, pad_h:pad_h + input_shape[1], pad_w:pad_w + input_shape[2], :
    ]
    return input_gradient, weight_gradient, bias_gradient

# Max pooling, reduserer str på data, viktig mønstre beholdes
def max_pool_forward(x):
    height = x.shape[1] // 2
    width = x.shape[2] // 2
    cropped = x[:, :height * 2, :width * 2, :]
    windows = cropped.reshape(x.shape[0], height, 2, width, 2, x.shape[3])
    windows = windows.transpose(0, 1, 3, 5, 2, 4)
    flattened = windows.reshape(x.shape[0], height, width, x.shape[3], 4)
    winners = flattened.argmax(axis=-1)
    return flattened.max(axis=-1), (winners, x.shape, height, width)

# Max pooling backward
def max_pool_backward(gradient, cache):
    winners, input_shape, height, width = cache
    input_gradient = np.zeros(input_shape, dtype=gradient.dtype)
    for index, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        input_gradient[:, row:height * 2:2, col:width * 2:2, :] = (
            gradient * (winners == index)
        )
    return input_gradient

# Initialisere parametrene til nettverket
def initialize_parameters(rng, n_inputs, n_classes):
    scale = np.sqrt(2.0 / (3 * 3 * n_inputs))
    return {
        "conv_weights": (rng.standard_normal((N_FILTERS, 1, 3, 3)) * scale).astype(np.float32),
        "conv_bias": np.zeros(N_FILTERS, dtype=np.float32),
        "dense_weights": (
            rng.standard_normal((N_FILTERS, n_classes)) * np.sqrt(2.0 / N_FILTERS)
        ).astype(np.float32),
        "dense_bias": np.zeros(n_classes, dtype=np.float32),
    }

# Forward pass, tar spektogrammet gjennom konvolusjonslaget, og beregner
# en score til slutt for hver klasse.
def forward(x, parameters):
    x = x[..., np.newaxis]
    convolution, conv_cache = conv_forward(
        x, parameters["conv_weights"], parameters["conv_bias"]
    )
    activated = np.maximum(convolution, 0)
    pooled, pool_cache = max_pool_forward(activated)
    pooled_shape = pooled.shape
    averages = pooled.mean(axis=(1, 2))
    logits = averages @ parameters["dense_weights"] + parameters["dense_bias"]
    return logits, (
        convolution,
        conv_cache,
        pool_cache,
        pooled_shape,
        pooled.shape[1] * pooled.shape[2],
        averages,
    )

# Beregner hvor feil modellen er, og gradienten som trengs for å forbedre
# under trening.
def softmax_loss(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()
    probabilities[np.arange(len(labels)), labels] -= 1
    probabilities /= len(labels)
    return float(loss), probabilities

# Går "bakover" gjennom nettverket, beregner hvordan modelens feil påvirkers
# av vektene og bias-verdiene, oppdateres under trening.
def backward(logit_gradient, cache, parameters):
    convolution, conv_cache, pool_cache, pooled_shape, pooled_area, averages = cache
    dense_weight_gradient = averages.T @ logit_gradient
    dense_bias_gradient = logit_gradient.sum(axis=0)
    average_gradient = logit_gradient @ parameters["dense_weights"].T
    pooled_gradient = np.broadcast_to(
        average_gradient[:, np.newaxis, np.newaxis, :] / pooled_area,
        pooled_shape,
    )
    activation_gradient = max_pool_backward(pooled_gradient, pool_cache)
    convolution_gradient = activation_gradient * (convolution > 0)
    _, conv_weight_gradient, conv_bias_gradient = conv_backward(
        convolution_gradient, conv_cache
    )
    return {
        "conv_weights": conv_weight_gradient,
        "conv_bias": conv_bias_gradient,
        "dense_weights": dense_weight_gradient,
        "dense_bias": dense_bias_gradient,
    }

# Forutsier hvilken klasse hvert spektogram tilhører
def predict(x, parameters, batch_size=32):
    predictions = []
    for start in range(0, len(x), batch_size):
        logits, _ = forward(x[start:start + batch_size], parameters)
        predictions.extend(logits.argmax(axis=1))
    return np.asarray(predictions, dtype=np.int64)

# Nøyaktighet
def accuracy(labels, predictions):
    return float(np.mean(labels == predictions))

# Confusion matrix
def confusion_matrix(labels, predictions, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix

# Skriver ut nøyaktighet,recall,F1-score og confusion matrix
def classification_report(labels, predictions):
    matrix = confusion_matrix(labels, predictions, len(CLASS_NAMES))
    print(f"{'klasse':12s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'antall':>10s}")
    for index, name in enumerate(CLASS_NAMES):
        true_positive = matrix[index, index]
        precision = true_positive / max(1, matrix[:, index].sum())
        recall = true_positive / max(1, matrix[index, :].sum())
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        print(
            f"{name:12s} {precision:10.3f} {recall:10.3f} "
            f"{f1:10.3f} {matrix[index, :].sum():10d}"
        )
    print("\nConfusion Matrix (rader=faktisk, kolonner=predikert):")
    print(matrix)

# Trener nettverket
def train_batch(x, labels, parameters, optimizer_state, step):
    logits, cache = forward(x, parameters)
    loss, logit_gradient = softmax_loss(logits, labels)
    gradients = backward(logit_gradient, cache, parameters)
    for name in ("conv_weights", "dense_weights"):
        gradients[name] += WEIGHT_DECAY * parameters[name]

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for name, parameter in parameters.items():
        first_moment, second_moment = optimizer_state[name]
        first_moment *= beta1
        first_moment += (1 - beta1) * gradients[name]
        second_moment *= beta2
        second_moment += (1 - beta2) * gradients[name] ** 2
        corrected_first = first_moment / (1 - beta1 ** step)
        corrected_second = second_moment / (1 - beta2 ** step)
        parameter -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + epsilon)
    return loss


def main():
    rng = np.random.default_rng(SEED)
    X_train, y_train, X_test, y_test, conditions = load_features()
    train_indices, validation_indices = stratified_split(y_train, VALIDATION_FRACTION)

    # Normalisere data
    mean = float(X_train[train_indices].mean())
    std = float(X_train[train_indices].std())
    if std == 0:
        raise ValueError("Treningsdataene har standardavvik 0; kan ikke normalisere.")
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    parameters = initialize_parameters(rng, 1, len(CLASS_NAMES))
    optimizer_state = {
        name: (np.zeros_like(parameter), np.zeros_like(parameter))
        for name, parameter in parameters.items()
    }
    best_parameters = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    step = 0

    print(
        f"NumPy-CNN fra scratch; {len(train_indices)} treningsklipp, "
        f"{len(validation_indices)} valideringsklipp."
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(train_indices)
        losses = []
        for start in range(0, len(train_indices), BATCH_SIZE):
            batch_indices = train_indices[start:start + BATCH_SIZE]
            step += 1
            losses.append(
                train_batch(
                    X_train[batch_indices],
                    y_train[batch_indices],
                    parameters,
                    optimizer_state,
                    step,
                )
            )

        validation_logits = []
        for start in range(0, len(validation_indices), BATCH_SIZE):
            logits, _ = forward(
                X_train[validation_indices[start:start + BATCH_SIZE]], parameters
            )
            validation_logits.append(logits)
        validation_logits = np.concatenate(validation_logits)
        validation_loss, _ = softmax_loss(validation_logits, y_train[validation_indices])
        validation_accuracy = accuracy(
            y_train[validation_indices], validation_logits.argmax(axis=1)
        )
        print(
            f"Epoch {epoch:03d}: train_loss={np.mean(losses):.4f}, "
            f"val_loss={validation_loss:.4f}, val_accuracy={validation_accuracy:.3f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_parameters = {
                name: parameter.copy() for name, parameter in parameters.items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping etter {epoch} epoker.")
                break

    if best_parameters is None:
        raise RuntimeError("Ingen beste modelltilstand ble lagret under treningen.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        **best_parameters,
        class_names=np.asarray(CLASS_NAMES),
        normalization_mean=np.asarray(mean),
        normalization_std=np.asarray(std),
    )
    print(f"\n Beste modellen lagret i {MODEL_PATH}")

    test_predictions = predict(X_test, best_parameters)
    print(f"\nTest accuracy: {accuracy(y_test, test_predictions):.3f}")
    classification_report(y_test, test_predictions)

    by_condition = defaultdict(list)
    for condition, actual, predicted in zip(conditions, y_test, test_predictions):
        by_condition[str(condition)].append(actual == predicted)
    print("\nTest accuracy per støyforhold:")
    for condition, results in sorted(by_condition.items()):
        print(f"{condition:15s} {np.mean(results):.3f} ({sum(results)}/{len(results)})")


if __name__ == "__main__":
    main()

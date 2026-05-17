from scipy.spatial.distance import cdist
import numpy as np

def match_features(train_features, train_labels, test_features, test_labels):
    """
    Calculates distances between test and train features and separates them
    into Genuine and Imposter scores.
    """
    distance_matrix = cdist(test_features, train_features, metric='euclidean')

    genuine_distances = []
    imposter_distances = []
    predictions = []

    for i in range(len(test_labels)):
        test_label = test_labels[i]

        closest_train_idx = np.argmin(distance_matrix[i])
        predicted_label = train_labels[closest_train_idx]
        predictions.append(predicted_label)

        for j in range(len(train_labels)):
            train_label = train_labels[j]
            distance = distance_matrix[i, j]

            if test_label == train_label:
                genuine_distances.append(distance)
            else:
                imposter_distances.append(distance)

    return predictions, np.array(genuine_distances), np.array(imposter_distances)


def calculate_rank1_accuracy(predictions, true_labels):
    """Calculates Rank-1 Accuracy (TPIR) as a percentage."""
    correct_matches = np.sum(np.array(predictions) == np.array(true_labels))
    total_tests = len(true_labels)
    accuracy = (correct_matches / total_tests) * 100
    return accuracy
# feature_extraction.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.feature import hog, local_binary_pattern

def extract_features_pca(X_train, X_test, n_components=50):
    """Applies PCA to the dataset."""
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


def extract_features_lda(X_train, y_train, X_test, n_components=39):
    """Applies LDA to the dataset dynamically."""

    # Calculate how many unique people are in the training set
    n_classes = len(np.unique(y_train))
    max_components = n_classes - 1

    # Dynamically cap the components so the math never breaks
    if n_components > max_components:
        actual_components = max_components
    else:
        actual_components = n_components

    # If max_components is 0 (only 1 person enrolled), LDA will fail.
    # Catch this gracefully.
    if actual_components <= 0:
        raise ValueError("LDA requires at least 2 enrolled subjects to work! Please enroll a second person.")

    lda = LDA(n_components=actual_components)

    # Notice LDA needs the labels (y_train) to fit
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    return X_train_lda, X_test_lda, lda



def extract_hog_lbp_fusion(X_data, image_shape=(112, 92)):
    """
    Takes flattened images, reshapes them, and extracts a fused HOG + LBP vector.
    """
    fused_features = []

    # LBP Parameters
    radius = 1
    n_points = 8 * radius

    for flattened_img in X_data:
        # 1. Reshape back to 2D image because HOG/LBP need spatial context
        img_2d = flattened_img.reshape(image_shape)

        # 2. Extract HOG (Shape/Edges)
        hog_vec = hog(img_2d, orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(1, 1), visualize=False)

        # 3. Extract LBP (Texture)
        lbp_img = local_binary_pattern(img_2d, n_points, radius, method='uniform')
        # Convert the LBP image into a histogram (feature vector)
        (lbp_hist, _) = np.histogram(lbp_img.ravel(),
                                     bins=np.arange(0, n_points + 3),
                                     range=(0, n_points + 2))

        # Normalize the LBP histogram so it doesn't overpower the HOG values
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)

        # 4. FEATURE FUSION: Concatenate HOG and LBP vectors
        fused_vec = np.hstack((hog_vec, lbp_hist))
        fused_features.append(fused_vec)

    return np.array(fused_features)


def train_fusion_model(X_train, y_train, X_test):
    """
    Extracts fused features and applies PCA to reduce the massive fused vector
    down to a manageable size before matching.
    """
    print("Extracting HOG+LBP features for Training data...")
    X_train_fused = extract_hog_lbp_fusion(X_train)

    print("Extracting HOG+LBP features for Testing data...")
    X_test_fused = extract_hog_lbp_fusion(X_test)

    # --- THE FIX: DYNAMIC PCA COMPONENTS ---
    # Calculate the maximum possible components we are mathematically allowed to extract
    n_samples = X_train_fused.shape[0]
    n_features = X_train_fused.shape[1]
    max_allowed = min(n_samples, n_features)

    # We want 50, but if we don't have enough images, we just use the max allowed.
    actual_components = min(50, max_allowed)

    print(f"Applying PCA (Shrinking to {actual_components} features)...")
    pca = PCA(n_components=actual_components)

    X_train_final = pca.fit_transform(X_train_fused)
    X_test_final = pca.transform(X_test_fused)

    return X_train_final, X_test_final, pca
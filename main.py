# main.py
import AT_database
import feature_extraction
import matching
import evaluation

def main():
    print("Loading AT&T dataset (Academic Benchmark)...")
    DATASET_PATH = "archive"
    X_train, y_train, X_test, y_test = AT_database.load_att_dataset(DATASET_PATH)

    print("Extracting features (PCA, LDA, & FUSION)...")
    pca_train, pca_test = feature_extraction.extract_features_pca(X_train, X_test)
    lda_train, lda_test, _ = feature_extraction.extract_features_lda(X_train, y_train, X_test)
    fusion_train, fusion_test, _ = feature_extraction.train_fusion_model(X_train, y_train, X_test)

    # --- PCA EVALUATION ---
    print("\n--- 1. PCA Results ---")
    pca_preds, pca_gen, pca_imp = matching.match_features(pca_train, y_train, pca_test, y_test)
    pca_accuracy = matching.calculate_rank1_accuracy(pca_preds, y_test)
    print(f"PCA Rank-1 Accuracy: {pca_accuracy:.2f}%")

    # 1. Trigger the Gen-Imp Curve Plot
    evaluation.plot_gen_imp(pca_gen, pca_imp, "PCA")

    # 2. Capture and print ALL required metrics (including TMR)
    pca_d, pca_eer, pca_tmr1, pca_tmr01 = evaluation.calculate_metrics(pca_gen, pca_imp, "PCA")
    print(
        f"PCA D-prime: {pca_d:.4f} | EER: {pca_eer:.4f} | TMR @ 1% FMR: {pca_tmr1:.4f} | TMR @ 0.01% FMR: {pca_tmr01:.4f}")

    # --- LDA EVALUATION ---
    print("\n--- 2. LDA Results ---")
    lda_preds, lda_gen, lda_imp = matching.match_features(lda_train, y_train, lda_test, y_test)
    lda_accuracy = matching.calculate_rank1_accuracy(lda_preds, y_test)
    print(f"LDA Rank-1 Accuracy: {lda_accuracy:.2f}%")

    # Trigger the Gen-Imp Curve Plot
    evaluation.plot_gen_imp(lda_gen, lda_imp, "LDA")

    # Capture and print ALL required metrics
    lda_d, lda_eer, lda_tmr1, lda_tmr01 = evaluation.calculate_metrics(lda_gen, lda_imp, "LDA")
    print(
        f"LDA D-prime: {lda_d:.4f} | EER: {lda_eer:.4f} | TMR @ 1% FMR: {lda_tmr1:.4f} | TMR @ 0.01% FMR: {lda_tmr01:.4f}")

    # --- FUSION EVALUATION ---
    print("\n--- 3. HOG+LBP Fusion Results ---")
    fusion_preds, fusion_gen, fusion_imp = matching.match_features(fusion_train, y_train, fusion_test, y_test)
    fusion_accuracy = matching.calculate_rank1_accuracy(fusion_preds, y_test)
    print(f"FUSION Rank-1 Accuracy: {fusion_accuracy:.2f}%")

    # Trigger the Gen-Imp Curve Plot
    evaluation.plot_gen_imp(fusion_gen, fusion_imp, "Fusion (HOG+LBP)")

    # Capture and print ALL required metrics
    fusion_d, fusion_eer, fusion_tmr1, fusion_tmr01 = evaluation.calculate_metrics(fusion_gen, fusion_imp, "Fusion")
    print(
        f"FUSION D-prime: {fusion_d:.4f} | EER: {fusion_eer:.4f} | TMR @ 1% FMR: {fusion_tmr1:.4f} | TMR @ 0.01% FMR: {fusion_tmr01:.4f}")

    print("\nAcademic Evaluation Complete.")
if __name__ == "__main__":
    main()
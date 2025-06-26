# Evaluate the model
print("\nEvaluation Metrics (Model on Top 15 Features):")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"AUC       : {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', annot_kws={"size": 14}, 
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])
plt.title("Confusion Matrix (Model + Top 15 Features)", fontsize=16)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix_XG.png", dpi=300, bbox_inches='tight')
plt.show()




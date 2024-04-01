

results_df <- read_tsv("/inwosu/imbalanced_learn/accuracy_file.txt", col_names = F) 
colnames(results_df) <- c("Dataset", "Resampling_strategy", "Model", "roc_auc_score", "Accuracy", "Precision", "Recall", "F1 score")


#graph 1
png("/inwosu/imbalanced_learn/Data/imb_learn.png", width = 18, height = 10, units = "in", res = 300)
results_df %>%
  ggplot(aes(x = factor(Resampling_strategy), y = roc_auc_score, fill = factor(Model))) +
  theme(text = element_text(size = 15)) +
  geom_jitter(color = "black", size = 0.3, alpha = 0.4, height = 0) + #test jitter
  geom_boxplot(alpha = 0.5, outlier.shape = NA) + 
  guides(fill = guide_legend(title = "Model")) +
  xlab("Resampling strategy")
dev.off()


#graph 2
png("/inwosu/imbalanced_learn/Data/imb_learn_2.png", width = 18, height = 10, units = "in", res = 300)
results_df %>%
  ggplot(aes(x = factor(Resampling_strategy), y = `roc_auc_score`, fill = factor(Model))) +
  theme(text = element_text(size = 15)) +
  geom_jitter(color = "black", size = 0.3, alpha = 0.4) +
  geom_boxplot(alpha = 0.5, outlier.shape = NA) + 
  facet_wrap(~Model) +
  #guides(fill = guide_legend(title = "Model")) +
  guides(fill = "none") +
  xlab("Resampling strategy")
dev.off()







# 
# ggsave("HER2_plot.pdf")
# 
# accuracy_file <- read_tsv("/inwosu/Meta_Analysis/accuracy_file_HER2.txt", col_names = F) 
# names(accuracy_file) <- c("Dataset_ID", "meta_gene", "meta_accuracy", "random_genes", "random_accuracy", "num_genes") 
# 
# new_file <- accuracy_file |>
#   mutate(new_col = ceiling(row_number() / 5)) |>
#   group_by(new_col, Dataset_ID, meta_accuracy, random_accuracy, num_genes) |>
#   summarise(meta = paste0(meta_gene, collapse = ","), random = paste0(random_genes, collapse = ",")) |>
#   ungroup()
# meta_unique <- unique(accuracy_file$meta_accuracy) |> tibble()
# 

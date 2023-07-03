
echo "single multi-target learning tasks."

bash 2.0.multi_target_learning_pipeline.sh >1204-2.0.log

echo "replay control results."

cd ../

bash 3.1.ablation_evaluation.sh > 1204-new_ablation.log

cd paraphrase


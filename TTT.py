from kbqa_classifier.model.stage1_bert_classifier import (
      load_stage1_model_and_tokenizer,
      predict_stage1,
  )
from kbqa_classifier.model.stage2_bert_classifier import (
      load_stage2_model_and_tokenizer,
      predict_stage2_bucket,
  )

question = "How many years were the Sumerian city-states at war?"

model1, tokenizer1 = load_stage1_model_and_tokenizer("output/stage1")
stage1_result = predict_stage1(question, model1, tokenizer1)[0]
print(stage1_result)

if stage1_result["label_id"] == 1:
    model2, tokenizer2 = load_stage2_model_and_tokenizer("output/stage2")
    stage2_result = predict_stage2_bucket(question, model2, tokenizer2)[0]
    print(stage2_result)
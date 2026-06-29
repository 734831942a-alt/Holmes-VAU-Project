import inspect
from transformers import TrainingArguments
print(inspect.getsource(TrainingArguments.__post_init__))
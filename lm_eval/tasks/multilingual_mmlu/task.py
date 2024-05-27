"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
from lm_eval.api.task import MultipleChoiceTask

_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""
LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')

def create_task(lang):
    class HendrycksTest(GeneralHendrycksTest):
        def __init__(self):
            super().__init__(lang)
    HendrycksTest.__name__ = f"MMLU_{lang}"

    return HendrycksTest


class GeneralHendrycksTest(MultipleChoiceTask):
    VERSION = 0
    NUM_FEW_SHOT = 25
    DATASET_PATH = "datasets/m_mmlu"
    DATASET_NAME = None
    PROMPT_TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    def __init__(self, lang):
        self.DATASET_NAME = f'mmlu_{lang}'
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            instruction = doc['question'] + "\n"
            instruction += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": doc["choices"],
            "gold": keys.index(doc["answer"])
            if isinstance(doc["answer"], str)
            else doc["answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
    
for lang in LANGS:
    globals()[f"MMLU_{lang}"] = create_task(lang)

# if __name__ == "__main__":
#     import yaml
#     LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')
#     for lang in LANGS:
#         task_dict = {
#             'task': f'mmlu_{lang}',
#             'class': f'!function task.MMLU_{lang}',
#         }
#         with open(f"mmlu_{lang}.yaml", "w") as f:
#             f.write(f'class: {task_dict["class"]}\n')
#             f.write(f'task: {task_dict["task"]}\n')
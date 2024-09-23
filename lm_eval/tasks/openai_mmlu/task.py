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

LANGS = ["mmlu_AR-XY",
            "mmlu_BN-BD",
            "mmlu_DE-DE",
            "mmlu_EN-US",
            "mmlu_ES-LA",
            "mmlu_FR-FR",
            "mmlu_HI-IN",
            "mmlu_ID-ID",
            "mmlu_IT-IT",
            "mmlu_JA-JP",
            "mmlu_KO-KR",
            "mmlu_PT-BR",
            "mmlu_ZH-CN",
            "mmlu_SW-KE",
            "mmlu_YO-NG",
            ]
def create_task(lang):
    class HendrycksTest(OpenAIHendrycksTest):
        def __init__(self):
            super().__init__(lang)
    HendrycksTest.__name__ = f"MMLU_{lang}"

    return HendrycksTest


class OpenAIHendrycksTest(MultipleChoiceTask):
    VERSION = 1
    NUM_FEW_SHOT = 5
    DATASET_PATH = "openai/MMMLU"
    DATASET_NAME = "by_language"
    PROMPT_TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    A_TEMPLATE = {
        "en": "Answer: ",
        "es": "Respuesta: ",
        "ru": "Ответ: ",
        "zh": "答案：",
    }
    def __init__(self, lang):
        # self.DATASET_NAME = f'mmlu_{lang}'
        self.lang = lang
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            instruction = doc['Question'] + "\n"
            instruction += "".join(
                # [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
                [f"{key}. {doc[f'{key}']}" for key in keys]
            )
            prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
            prompt += self.A_TEMPLATE.get(self.lang)
            print(prompt)
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": keys.index(doc["Answer"])
            if isinstance(doc["Answer"], str)
            else doc["Answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        # return doc["query"]
        return self._process_doc(doc)["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
    
for lang in LANGS:
    globals()[f"openai_mmlu_{lang}"] = create_task(lang)

if __name__ == "__main__":
    import yaml
    LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')
    for lang in LANGS:
        task_dict = {
            'task': f'mmlu_{lang}',
            'class': f'!function task.MMLU_{lang}',
        }
        with open(f"openai_mmlu_{lang}.yaml", "w") as f:
            f.write(f'class: {task_dict["class"]}\n')
            f.write(f'task: {task_dict["task"]}\n')
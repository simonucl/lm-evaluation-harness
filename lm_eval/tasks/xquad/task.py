"""
Know What You Don’t Know: Unanswerable Questions for SQuAD
https://arxiv.org/pdf/1806.03822.pdf

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
consisting of questions posed by crowdworkers on a set of Wikipedia articles,
where the answer to every question is a segment of text, or span, from the
corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable
questions written adversarially by crowdworkers to look similar to answerable ones.
To do well on SQuAD2.0, systems must not only answer questions when possible, but
also determine when no answer is supported by the paragraph and abstain from answering.

Homepage: https://rajpurkar.github.io/SQuAD-explorer/
"""
import datasets
from math import exp
from functools import partial
from packaging import version
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask

_CITATION = """
@misc{rajpurkar2018know,
    title={Know What You Don't Know: Unanswerable Questions for SQuAD},
    author={Pranav Rajpurkar and Robin Jia and Percy Liang},
    year={2018},
    eprint={1806.03822},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric("squad")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references).get(key, 0)


class XQuAD(ConfigurableTask):
    VERSION = 1
    DATASET_PATH = "xquad"
    DATASET_NAME = None
    NUM_FEW_SHOT = 0
    PROMPT_TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    INSTRUCTION = "Answer the question in the input based on the given context. Your answer should be directly extracted from the context, and it should be a single entity, name, or number, not a sentence."
    Q = "Question: "
    C = "Context: "
    A = "Answer: "
    # # HF changed squad on us so we have to make sure we aren't running the old one
    # assert version.parse(datasets.__version__) >= version.parse(
    #     "1.11.0"
    # ), "datasets v1.11.0 or later required for SQuAD"

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    # def training_docs(self):
    #     return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # input = self.Q + doc["question"] + "\n" + self.C + doc["context"]
        input = self.C + doc["context"] + "\n" + self.Q + doc["question"]
        prompt = self.PROMPT_TEMPLATE.format(instruction=self.INSTRUCTION, input=input) + self.A
        return prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        if len(answer_list) > 0:
            answer = answer_list[0]
        return " " + answer

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
            **kwargs,
        )

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation = results
        # continuation, (logprob_unanswerable, _) = results

        # no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
            # "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "exact": partial(
                _squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }

class XQuAD_es(XQuAD):
    DATASET_NAME = "xquad.es"
    INSTRUCTION = "Responde a la pregunta en el input basándote en el contexto dado. Tu respuesta debe ser extraída directamente del contexto, y debe ser una entidad, nombre o número, no una oración."
    Q = "Pregunta: "
    C = "Contexto: "
    A = "Respuesta: "

class XQuAD_de(XQuAD):
    DATASET_NAME = "xquad.de"
    INSTRUCTION = "Beantworte die Frage im Input basierend auf dem gegebenen Kontext. Deine Antwort sollte direkt aus dem Kontext extrahiert werden und sollte eine einzelne Entität, einen Namen oder eine Zahl sein, nicht ein Satz."
    Q = "Frage: "
    C = "Kontext: "
    A = "Antwort: "

class XQuAD_el(XQuAD):
    DATASET_NAME = "xquad.el"
    INSTRUCTION = "Απάντησε στην ερώτηση στην είσοδο βασιζόμενος στον δεδομένο πλαίσιο. Η απάντησή σας πρέπει να εξαχθεί απευθείας από το πλαίσιο και πρέπει να είναι μια μόνο οντότητα, όνομα ή αριθμός, όχι μια πρόταση."
    Q = "Ερώτηση: "
    C = "Πλαίσιο: "
    A = "Απάντηση: "

class XQuAD_ru(XQuAD):
    DATASET_NAME = "xquad.ru"
    INSTRUCTION = "Ответьте на вопрос во входных данных на основе предоставленного контекста. Ваш ответ должен быть непосредственно извлечен из контекста и должен быть одной сущностью, именем или числом, а не предложением."
    Q = "Вопрос: "
    C = "Контекст: "
    A = "Ответ: "

class XQuAD_tr(XQuAD):
    DATASET_NAME = "xquad.tr"
    INSTRUCTION = "Girişteki bağlamı dikkate alarak soruya yanıt verin. Yanıtınız, bağlamdan doğrudan çıkarılmalı ve bir varlık, ad veya sayı olmalı, cümle olmamalıdır."
    Q = "Soru: "
    C = "Bağlam: "
    A = "Yanıt: "

class XQuAD_ar(XQuAD):
    DATASET_NAME = "xquad.ar"
    INSTRUCTION = "أجب على السؤال في الإدخال استنادًا إلى السياق المعطى. يجب أن تكون إجابتك مستخرجة مباشرة من السياق، ويجب أن تكون كيانًا واحدًا أو اسمًا أو رقمًا، وليس جملة."
    Q = "سؤال: "
    C = "السياق: "
    A = "الإجابة: "

class XQuAD_vi(XQuAD):
    DATASET_NAME = "xquad.vi"
    INSTRUCTION = "Trả lời câu hỏi trong đầu vào dựa trên ngữ cảnh đã cho. Câu trả lời của bạn phải được trích xuất trực tiếp từ ngữ cảnh và nó phải là một thực thể, tên hoặc số, không phải là một câu."
    Q = "Câu hỏi: "
    C = "Bối cảnh: "
    A = "Câu trả lời: "

class XQuAD_th(XQuAD):
    DATASET_NAME = "xquad.th"
    INSTRUCTION = "ตอบคำถามในข้อมูลนำเข้าโดยพิจารณาจากบริบทที่กำหนดให้ คำตอบของคุณควรถูกดึงมาจากบริบทโดยตรง และควรเป็นสิ่งเดียว ชื่อหรือตัวเลข ไม่ใช่ประโยค"
    Q = "คำถาม: "
    C = "บริบท: "
    A = "คำตอบ: "

class XQuAD_zh(XQuAD):
    DATASET_NAME = "xquad.zh"
    INSTRUCTION = "根据给定的上下文回答输入中的问题。您的答案应直接从上下文中提取，应为单个实体、名称或数字，而不是句子。"
    Q = "问题: "
    C = "上下文: "
    A = "答案: "

class XQuAD_hi(XQuAD):
    DATASET_NAME = "xquad.hi"
    INSTRUCTION = "दिए गए संदर्भ पर आधारित प्रश्न का उत्तर दें। आपका उत्तर सीधे संदर्भ से निकाला जाना चाहिए, और यह एक एकल इकाई, नाम या संख्या होना चाहिए, न कि एक वाक्य।"
    Q = "प्रश्न: "
    C = "संदर्भ: "
    A = "उत्तर: "

class XQuAD_en(XQuAD):
    DATASET_NAME = "xquad.en"

# LANG_CLASSES = [
#     XQuAD_Arabic,
#     XQuAD_Chinese,
#     XQuAD_German,
#     XQuAD_Greek,
#     XQuAD_Hindi,
#     XQuAD_Russian,
#     XQuAD_Spanish,
#     XQuAD_Thai,
#     XQuAD_Turkish,
#     XQuAD_Vietnamese,
#     XQuAD_English,
# ]

# def construct_tasks():
#     tasks = {}
#     for lang, lang_class in zip(
#         ["ar", "zh", "de", "el", "hi", "ru", "es", "th", "tr", "vi", "en"],
#         LANG_CLASSES,
#     ):
#         tasks[f"xquad_{lang}"] = lang_class
#     return tasks

# if __name__ == "__main__":
#     # iterate over the langs and save it into a yaml file, with task: xquad_{lang} and class: !function task.XQuAD_{lang}
#     import yaml
#     # tasks = construct_tasks()
#     languages = ["ar", "zh", "de", "el", "hi", "ru", "es", "th", "tr", "vi", "en"]
#     task_dict = {}
#     for lang in languages:
#         task_dict['task'] = f"xquad_{lang}"
#         task_dict['class'] = f"!function task.XQuAD_{lang}"
#         with open(f"xquad_{lang}.yaml", "w") as f:
#             yaml.dump(task_dict, f, default_flow_style=False)

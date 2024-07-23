from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("lowercase")
class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("uppercase")
class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("map")
class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = None, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        if mapping_dict is None:
            mapping_dict = {}
        assert isinstance(
            mapping_dict, dict
        ), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("format_span")
class SAPNFilter(Filter):
    def __init__(self, pos=False) -> None:
        self.pos = pos

    def apply(self, resps, docs):
        def format_ner_text(text, target=False):
            label_dict = {'person': 'PER',
                          'location': 'LOC',
                          'organization': 'ORG',
                          'miscellaneous': 'MISC',
                          'date': 'DATE',
                          'per': 'PER',
                          'loc': 'LOC',
                          'org': 'ORG',
                          'misc': 'MISC'}
            text = text.lower()
            for key, value in label_dict.items():
                text = text.replace(key, value) if not target else text.replace(value.lower(), value)

            text = "$".join(i for i in text.split('$$'))
            return text.rstrip('$$')

        def format_pos_text(text, target=False):
            label_dict = {'adjectives': 'ADJ',
                          'adposition': 'ADP',
                          'adverbs': 'ADV',
                          'auxiliary': 'AUX',
                          'conjunctions': 'CONJ',
                          'coordinating conjunction': 'CCONJ',
                          'subordinating conjunction conjunction': 'SCONJ',
                          'determiners': 'DET',
                          'interjections': 'INTJ',
                          'nouns': 'NOUN',
                          'numeral': 'NUM',
                          'particle': 'PART',
                          'pronouns': 'PRON',
                          'prepositions': 'LOC',
                          'proper noun': 'PROPN',
                          'punctuation': 'PUNCT',
                          'verb': 'VERB',
                          'symbol': 'SYM',
                          'other': 'X'
                          }
            text = text.lower()
            for key, value in label_dict.items():
                text = text.replace(key, value) if not target else text.replace(value.lower(), value)

            text = "$".join(i for i in text.split('$$'))
            return text.rstrip('$$')

        def filter_set(inst):
            if self.pos:
                return [format_pos_text(resp.lower()) for resp in inst]
            else:
                return [format_ner_text(resp.lower()) for resp in inst]
        return [filter_set(resp) for resp in resps]

from haystack.nodes import PreProcessor


class SplitCleanerPreProcessor(PreProcessor):
    def __init__(self, *args, split_cleaner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_cleaner = split_cleaner

    def _split_into_units(self, text, split_by):
        units, sep = super()._split_into_units(text, split_by)
        if self.split_cleaner:
            units = self.split_cleaner(units)
        return units, sep

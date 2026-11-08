from condition.numeric_condition import NumericConditionFactory
from condition.set_condition import SetConditionFactory
from condition.categorical_condition import CategoricalConditionFactory
from condition.text_condition import TextConditionFactory


class ProcessingService:
    def __init__(self, condition_factories):
        self._condition_factories = condition_factories

    @classmethod
    def create_default(cls):
        numeric_factory = NumericConditionFactory(max_thresholds=20)
        set_factory = SetConditionFactory(min_support=2, max_candidates=25)
        categorical_factory = CategoricalConditionFactory(min_support=2, max_candidates=20)
        text_factory = TextConditionFactory(min_support=2, max_candidates=15)

        default_factories = {
            'genres': set_factory,
            'cast_list': set_factory,
            'keywords': set_factory,
            'production_companies': set_factory,
            'release_year': numeric_factory,
            'vote_average': numeric_factory,
            'runtime': numeric_factory,
            'title': text_factory,
            'director': categorical_factory,
            'original_language': categorical_factory,
            'collection_name': categorical_factory,
        }

        return cls(condition_factories=default_factories)

    def prepare_features_and_factories(self, df_raw):
        df = df_raw.set_index('movie_id') if 'movie_id' in df_raw.columns else df_raw.copy()
        active_factories = {
            comp: factory
            for comp, factory in self._condition_factories.items()
            if comp in df.columns
        }
        print(f"Przygotowano {len(df)} filmów oraz {len(active_factories)} fabryk warunków dla drzew decyzyjnych...")
        return df, active_factories

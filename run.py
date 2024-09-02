from src.options import *
from src.model_io import *

def main():
    a = load_atlas_model(
        'data/models/atlas/base',
        Options.from_dict(
            name = 'some_experiment',
            generation_max_length = 32,
            target_maxlength = 32,
            gold_score_mode = "ppmean",
            precision = 'bf16',
            reader_model_type = 'google/t5-base-lm-adapt',
            text_maxlength = 512,
            model_path = 'data/models/atlas/base',
            load_index_path = 'data/models/atlas/wiki/base',
            eval_data = 'data/data/',
            per_gpu_batch_size = 1,
            n_context = 40,
            retriever_n_context = 40,
            index_mode = "flat",
            task = "qa",
            write_results = True,
        )
    )

if __name__ == '__main__':
    main()

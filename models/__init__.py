from .cbrain import build_cbrain
from .cbrain_purefmri import build_cbrain_purefmri
from .random_guess import build_random_guess
from .svm_hierarchy import build_svm_hierarchy

MODEL_FUNCS = {
    "cbrain": build_cbrain,
    "cbrain_purefmri": build_cbrain_purefmri,
    "random_guess": build_random_guess,
    "svm_hierarchy": build_svm_hierarchy,
}

def build_model(args, dataset_config):
    model = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model
def compare_algo(df, target, exp):
    exp.setup(df, target = target, use_gpu=True)
    best = exp.compare_models()
    best_model = exp.pull()
    return best_model
import wesep_real_tse.wesep.models.bsrnn_legacy as bsrnn_legacy


def get_model(model_name: str):
    if model_name.startswith("BSRNN"):
        return getattr(bsrnn_legacy, model_name)
    else:
        print(model_name + " not found !!!")
        exit(1)
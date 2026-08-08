import wespeaker.models.ecapa_tdnn as ecapa_tdnn


def get_speaker_model(model_name: str):
    if model_name.startswith("ECAPA_TDNN"):
        return getattr(ecapa_tdnn, model_name)
    else:
        print(model_name + " not found !!!")
        exit(1)
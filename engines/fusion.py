def confidence_weighted_fusion(
    audio_state,
    text_state,
    audio_confidence=0.7,
    text_confidence=0.7
):

    total = (
        audio_confidence
        +
        text_confidence
    )

    if total <= 0:

        audio_weight = 0.5
        text_weight = 0.5

    else:

        audio_weight = (
            audio_confidence / total
        )

        text_weight = (
            text_confidence / total
        )

    return {

        "valence":
        (
            audio_weight
            *
            audio_state.get(
                "valence",
                0.0
            )
        )
        +
        (
            text_weight
            *
            text_state.get(
                "valence",
                0.0
            )
        ),

        "arousal":
        (
            audio_weight
            *
            audio_state.get(
                "arousal",
                0.0
            )
        )
        +
        (
            text_weight
            *
            text_state.get(
                "arousal",
                0.0
            )
        ),

        "audio_weight":
        audio_weight,

        "text_weight":
        text_weight
    }
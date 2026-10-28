"""Explicit task mappings; never collapse a three-class prediction into two."""
THREE_LABELS = ('KHONG_CANH_BAO', 'CAN_XAC_MINH', 'CANH_BAO')
BINARY_LABELS = ('KHONG_CANH_BAO', 'CANH_BAO')


def labels_for_mode(mode):
    if mode == 'three_class':
        return THREE_LABELS
    if mode == 'binary':
        return BINARY_LABELS
    raise ValueError(f'Unsupported classification_mode: {mode}')


def validate_mapping(config, mode):
    labels = labels_for_mode(mode)
    ids = {int(k): v for k, v in config.id2label.items()}
    if (ids != dict(enumerate(labels)) or
            config.label2id != {v: i for i, v in enumerate(labels)} or
            config.num_labels != len(labels)):
        raise ValueError(f'Checkpoint label mapping does not match {mode}')
    return labels

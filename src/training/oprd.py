from model.transplant.representation_alignment import representation_loss


def representation_distillation(student_hidden, specialist_hidden):
    return representation_loss(student_hidden, specialist_hidden, "normalized")


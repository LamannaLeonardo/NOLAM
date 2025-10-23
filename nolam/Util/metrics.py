import copy

from nolam.algorithm.ActionModel import ActionModel
import numpy as np

def action_model_statistics(evaluated_model, gt_model):

    eval_model = ActionModel(input_file=evaluated_model)
    gt_model = ActionModel(input_file=gt_model)

    for op_gt, op_eval in zip(gt_model.operators, eval_model.operators):
        op_gt.operator_name = op_gt.operator_name.replace('_', '-')
        op_eval.operator_name = op_eval.operator_name.replace('_', '-')

    # Sort operators list
    gt_model.operators = sorted(gt_model.operators, key=lambda x: x.operator_name, reverse=True)
    eval_operatos_name = [op.operator_name for op in eval_model.operators]
    sorted_eval_operators = []
    for op in gt_model.operators:
        if op.operator_name in eval_operatos_name:
            operator = next((o for o in eval_model.operators if o.operator_name == op.operator_name), None)
            sorted_eval_operators.append(operator)
        else:
            new_operator = copy.deepcopy(op)
            new_operator.precs_pos = []
            new_operator.precs_neg = []
            new_operator.eff_pos = []
            new_operator.eff_neg = []
            sorted_eval_operators.append(new_operator)
    eval_model.operators = sorted_eval_operators

    assert bool(np.all([gt_op.operator_name == learned_op.operator_name for gt_op, learned_op in zip(gt_model.operators, eval_model.operators)]))

    all_pre_pos_precision = []
    all_pre_neg_precision = []
    all_eff_pos_precision = []
    all_eff_neg_precision = []
    all_pre_pos_recall = []
    all_pre_neg_recall = []
    all_eff_pos_recall = []
    all_eff_neg_recall = []
    all_recall = []
    all_precision = []

    for gt_op, learned_op in zip(gt_model.operators, eval_model.operators):
        fn_pre_pos = len(set([p for p in gt_op.precs_pos if p not in learned_op.precs_pos]))
        fp_pre_pos = len(set([p for p in learned_op.precs_pos if p not in gt_op.precs_pos]))
        tp_pre_pos = len(set([p for p in learned_op.precs_pos if p in gt_op.precs_pos]))

        fn_pre_neg = len(set([p for p in gt_op.precs_neg if p not in learned_op.precs_neg]))
        fp_pre_neg = len(set([p for p in learned_op.precs_neg if p not in gt_op.precs_neg]))
        tp_pre_neg = len(set([p for p in learned_op.precs_neg if p in gt_op.precs_neg]))

        fn_eff_pos = len(set([p for p in gt_op.eff_pos if p not in learned_op.eff_pos]))
        fp_eff_pos = len(set([p for p in learned_op.eff_pos if p not in gt_op.eff_pos]))
        tp_eff_pos = len(set([p for p in learned_op.eff_pos if p in gt_op.eff_pos]))

        fn_eff_neg = len(set([p for p in gt_op.eff_neg if p not in learned_op.eff_neg]))
        fp_eff_neg = len(set([p for p in learned_op.eff_neg if p not in gt_op.eff_neg]))
        tp_eff_neg = len(set([p for p in learned_op.eff_neg if p in gt_op.eff_neg]))

        all_tp = tp_pre_pos + tp_eff_pos + tp_eff_neg + tp_pre_neg
        all_fp = fp_pre_pos + fp_pre_neg + fp_eff_pos + fp_eff_neg
        all_fn = fn_pre_pos + fn_pre_neg + fn_eff_pos + fn_eff_neg

        pre_pos_recall = pre_pos_precision = 0
        pre_neg_recall = pre_neg_precision = 0
        eff_pos_recall = eff_pos_precision = 0
        eff_neg_recall = eff_neg_precision = 0

        if tp_pre_pos + fp_pre_pos > 0:
            pre_pos_precision = tp_pre_pos / (tp_pre_pos + fp_pre_pos)

        if tp_pre_neg + fp_pre_neg > 0:
            pre_neg_precision = tp_pre_neg / (tp_pre_neg + fp_pre_neg)

        if tp_eff_pos + fp_eff_pos > 0:
            eff_pos_precision = tp_eff_pos / (tp_eff_pos + fp_eff_pos)

        if tp_eff_neg + fp_eff_neg > 0:
            eff_neg_precision = tp_eff_neg / (tp_eff_neg + fp_eff_neg)

        if tp_pre_pos + fn_pre_pos > 0:
            pre_pos_recall = tp_pre_pos / (tp_pre_pos + fn_pre_pos)

        if tp_pre_neg + fn_pre_neg > 0:
            pre_neg_recall = tp_pre_neg / (tp_pre_neg + fn_pre_neg)

        if tp_eff_pos + fn_eff_pos > 0:
            eff_pos_recall = tp_eff_pos / (tp_eff_pos + fn_eff_pos)

        if tp_eff_neg + fn_eff_neg > 0:
            eff_neg_recall = tp_eff_neg / (tp_eff_neg + fn_eff_neg)

        if all_tp + all_fp > 0:
            all_precision.append(all_tp / (all_tp + all_fp))

        if all_tp + all_fn > 0:
            all_recall.append(all_tp / (all_tp + all_fn))

        all_pre_pos_precision.append(pre_pos_precision)
        all_pre_neg_precision.append(pre_neg_precision)
        all_eff_pos_precision.append(eff_pos_precision)
        all_eff_neg_precision.append(eff_neg_precision)
        all_pre_pos_recall.append(pre_pos_recall)
        all_pre_neg_recall.append(pre_neg_recall)
        all_eff_pos_recall.append(eff_pos_recall)
        all_eff_neg_recall.append(eff_neg_recall)

    pre_pos_recall = np.mean(all_pre_pos_recall)
    pre_pos_precision = np.mean(all_pre_pos_precision)
    pre_neg_precision = np.mean(all_pre_neg_precision)
    pre_neg_recall = np.mean(all_pre_neg_recall)
    eff_pos_recall = np.mean(all_eff_pos_recall)
    eff_pos_precision = np.mean(all_eff_pos_precision)
    eff_neg_precision = np.mean(all_eff_neg_precision)
    eff_neg_recall = np.mean(all_eff_neg_recall)
    overall_precision = np.mean(all_precision)
    overall_recall = np.mean(all_recall)

    return pre_pos_recall, pre_neg_recall, eff_pos_recall, eff_neg_recall, \
        pre_pos_precision, pre_neg_precision, eff_pos_precision, eff_neg_precision, overall_recall, overall_precision


def eval_model(learned_model, gt_model):
    pre_pos_recall, pre_neg_recall, eff_pos_recall, eff_neg_recall, \
        pre_pos_precision, pre_neg_precision, eff_pos_precision, eff_neg_precision, overall_recall, \
        overall_precision = action_model_statistics(learned_model, gt_model)

    evaluate = {'Precs_pos_recall': "{0:.2f}".format(pre_pos_recall),
                'Precs_neg_recall': "{0:.2f}".format(pre_neg_recall),
                'Pos_recall': "{0:.2f}".format(eff_pos_recall),
                'Neg_recall': "{0:.2f}".format(eff_neg_recall),
                'Precs_pos_precision': "{0:.2f}".format(pre_pos_precision),
                'Precs_neg_precision': "{0:.2f}".format(pre_neg_precision),
                'Pos_precision': "{0:.2f}".format(eff_pos_precision),
                'Neg_precision': "{0:.2f}".format(eff_neg_precision),
                'Tot_recall': "{0:.2f}".format(overall_recall),
                'Tot_precision': "{0:.2f}".format(overall_precision)
                }
    return evaluate
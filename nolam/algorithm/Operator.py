
class Operator:

    def __init__(self, operator_name, parameters, precs_pos, precs_neg, eff_pos, eff_neg):

        self.operator_name = operator_name
        self.parameters = parameters
        self.precs_pos = precs_pos
        self.precs_neg = precs_neg
        self.eff_pos = eff_pos
        self.eff_neg = eff_neg

    def __str__(self):
        return f"{self.operator_name}({','.join(self.parameters)})"

# 导入pymoo模块
from pymoo.problems import get_problem
from pymoo.model.problem import Problem

# 创建一个DTLZ1问题的实例，指定变量数和目标数
problem = get_problem("dtlz1", n_var=7, n_obj=2)

# 定义一个转换后的问题类，继承自原始问题类
class TransformedProblem(Problem):

    # 初始化方法，接收原始问题作为参数，并定义问题的属性
    def __init__(self, problem):
        super().__init__(n_var=problem.n_var,
                         n_obj=problem.n_obj,
                         n_constr=problem.n_constr,
                         xl=problem.xl,
                         xu=problem.xu)
        # 将原始问题保存为类的属性
        self.problem = problem

    # 评估方法，接收决策变量x和输出字典out，并计算目标值和约束值
    def _evaluate(self, x, out, *args, **kwargs):
        # 调用原始问题的评估方法，得到原始目标值和约束值
        self.problem._evaluate(x, out, *args, **kwargs)

        # 将原始目标值取负数，得到转换后的目标值，并赋值给输出字典的"F"键
        out["F"] = -out["F"]

# 创建一个转换后的问题实例，传入原始问题作为参数
transformed_problem = TransformedProblem(problem)

# 获取转换后的问题的真实帕累托前沿（即最大值的前沿）
pf = transformed_problem.pareto_front()

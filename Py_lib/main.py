from __future__ import annotations
import numpy as np
from dataclasses import dataclass,field
from enum import IntFlag,IntEnum,auto


@dataclass
class Matrix:
    rows: int
    cols: int
    data: np.ndarray

class MVFlag(IntFlag):
    NONE = 0
    REQUIRES_GRAD = 1 << 0
    PARAMETER = 1 << 1
    INPUT = 1 << 2
    OUTPUT = 1 << 3
    DESIRED_OUTPUT = 1 << 4
    COST = 1 << 5

class MVOp(IntEnum):
    NULL = 0
    CREATE = auto()

    UNARY_START = auto()

    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()

    BINARY_START = auto()

    ADD = auto()
    SUB = auto()
    MATMUL = auto()
    CROSS_ENTROPY = auto()

MODEL_VAR_MAX_INPUTS = 2

def mv_num_inputs(op:MVOp) -> int:
    if op < MVOp.UNARY_START:
        return 0
    if op < MVOp.BINARY_START:
        return 1
    return 2

@dataclass
class ModelVar:
    index: int
    flags: MVFlag

    val: Matrix
    grad: Matrix | None = None
    op: MVOp = MVOp.NULL
    inputs: list[ModelVar | None] = field(
        default_factory=lambda:[None] * MODEL_VAR_MAX_INPUTS
    )

@dataclass
class ModelProgram:
    vars: list[ModelVar] = field(default_factory=list)
    size: int = 0

@dataclass
class ModelContext:
    num_vars: int = 0
    input: ModelVar | None = None
    output: ModelVar | None = None
    desired_output: ModelVar | None = None
    cost: ModelVar | None = None

    forward_prog: ModelProgram = field(default_factory=ModelProgram)
    cost_prog: ModelProgram = field(default_factory=ModelProgram)

@dataclass
class ModelTrainingDesc:
    train_images: Matrix
    train_labels: Matrix
    test_images: Matrix
    test_labels: Matrix

    epochs: int
    batch_size: int
    learning_rate: float

def draw_mnist_digit(data: np.ndarray) -> None:
    img = data.reshape(28, 28)
    for y in range(0, 28, 2):
        for x in range(28):
            top = 232 + int(img[y, x] * 23)
            bottom = 232 + int(img[y + 1, x] * 23)
            print(f"\x1b[38;5;{top}m\x1b[48;5;{bottom}m▀", end="")
        print("\x1b[0m")
    print("\x1b[0m", end="")


class Operations:
    def __init__(self,rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows,cols), dtype=np.float32)

    @staticmethod
    def mat_create(rows: int, cols: int) -> Matrix:
        return Matrix(rows,cols,np.zeros((rows,cols),dtype=np.float32))

    @staticmethod
    def mat_copy(dst: Matrix, src:Matrix) -> bool:
        if dst.rows != src.rows or dst.cols != src.cols:
            return False
        dst.data[:] = src.data
        return True
        
    @staticmethod
    def mat_clear(matrix: 'Matrix'):
        matrix.data.fill(0)

    @staticmethod
    def mat_fill(matrix: 'Matrix', value: float) -> None:
        matrix.data.fill(value)
        
    @staticmethod
    def mat_scale( matrix: 'Matrix', scalar: float) -> None:
        matrix.data *= scalar

    @staticmethod
    def mat_sum(matrix:'Matrix') -> float:
        return float(matrix.data.sum())

    @staticmethod
    def mat_argmax(matrix:'Matrix') -> int:
        return int(matrix.data.argmax())

    @staticmethod
    def mat_add(out:'Matrix',a:'Matrix',b:'Matrix') -> bool:
        if a.rows != b.rows or a.cols != b.cols:
            return False
        if out.rows != a.rows or out.cols != a.cols:
            return False
        np.add(a.data, b.data, out=out.data)

        return True

    @staticmethod
    def mat_sub(out:'Matrix', a:'Matrix',b:'Matrix') -> bool:
        if a.rows != b.rows or a.cols != b.cols:
            return False
        if out.rows != a.rows or out.cols != a.cols:
            return False
        np.subtract(a.data, b.data, out=out.data)
        
        return True

    @staticmethod
    def mat_mul_nn(out:Matrix ,a: Matrix, b: Matrix) -> None:
        out.data += a.data @ b.data

    @staticmethod
    def mat_mul_nt(out: Matrix,a: Matrix, b:Matrix) -> None:
        out.data += a.data @ b.data.T

    @staticmethod
    def mat_mul_tn(out: Matrix,a: Matrix, b:Matrix) -> None:
        out.data += a.data.T @ b.data

    @staticmethod
    def mat_mul_tt(out: Matrix,a: Matrix, b:Matrix) -> None:
        out.data += a.data.T @ b.data.T

    @staticmethod
    def mat_mul(out:Matrix,a:Matrix,b:Matrix,
                zero_out:bool,transpose_a:bool,transpose_b:bool) -> bool:
        a_rows = a.cols if transpose_a else a.rows
        a_cols = a.rows if transpose_a else a.cols
        b_rows = b.cols if transpose_b else b.rows
        b_cols = b.rows if transpose_b else b.cols

        if a_cols != b_rows:
            return False
        if out.rows != a_rows or out.cols != b_cols:
            return False

        if zero_out:
            Operations.mat_clear(out)

        match(transpose_a,transpose_b):
            case(False,False): Operations.mat_mul_nn(out,a,b)
            case(False,True): Operations.mat_mul_nt(out,a,b)
            case(True,False): Operations.mat_mul_tn(out,a,b)
            case(True,True): Operations.mat_mul_tt(out,a,b)

        return True

    @staticmethod
    def mat_relu(out: Matrix, input_matrix: Matrix) -> bool:
        if out.rows != input_matrix.rows or out.cols != input_matrix.cols:
            return False
        np.maximum(input_matrix.data, 0, out=out.data)
        return True

    @staticmethod
    def mat_sigmoid(out:Matrix,input_matrix:Matrix) -> bool:
        if out.rows != input_matrix.rows or out.cols != input_matrix.cols:
            return False
        np.negative(input_matrix.data, out=out.data)
        np.exp(out.data, out=out.data)
        np.add(out.data, 1.0, out=out.data)
        np.reciprocal(out.data, out=out.data)
        return True

    @staticmethod
    def mat_tanh(out:Matrix,input_matrix:Matrix) -> bool:
        if out.rows != input_matrix.rows or out.cols != input_matrix.cols:
            return False
        np.tanh(input_matrix.data,out=out.data)
        return True

    @staticmethod
    def mat_softmax(out:Matrix,input_matrix:Matrix) -> bool:
        if out.rows != input_matrix.rows or out.cols != input_matrix.cols:
            return False
        np.subtract(input_matrix.data, input_matrix.data.max(), out=out.data)
        np.exp(out.data, out=out.data)
        out.data /= out.data.sum()
        return True

    @staticmethod
    def mat_cross_entropy(out:Matrix,p:Matrix,q:Matrix) -> bool:
        if p.rows != q.rows or p.cols != q.cols:
            return False

        if out.rows != p.rows or out.cols != p.cols:
            return False

        mask = p.data != 0.0
        out.data.fill(0.0)
        out.data[mask] = p.data[mask] * -np.log(q.data[mask])
        return True

    @staticmethod
    def mat_relu_add_grad(out:Matrix,input_matrix:Matrix,grad:Matrix) -> bool:
        if out.rows != input_matrix.rows or out.cols != input_matrix.cols:
            return False
        if out.rows != grad.rows or out.cols != grad.cols:
            return False

        out.data += grad.data * (input_matrix.data > 0.0)
        return True

    @staticmethod
    def mat_tanh_add_grad(out: Matrix, tanh_out: Matrix, grad: Matrix) -> bool:
        if out.rows != tanh_out.rows or out.cols != tanh_out.cols:
            return False
        if out.rows != grad.rows or out.cols != grad.cols:
            return False

        out.data += grad.data * (1.0 - tanh_out.data ** 2)
        return True

    @staticmethod
    def mat_sigmoid_add_grad(out: Matrix, sigmoid_out: Matrix, grad: Matrix) -> bool:
        if out.rows != sigmoid_out.rows or out.cols != sigmoid_out.cols:
            return False
        if out.rows != grad.rows or out.cols != grad.cols:
            return False

        out.data += grad.data * sigmoid_out.data * (1.0 - sigmoid_out.data)
        return True
    
    @staticmethod
    def mat_softmax_add_grad(out:Matrix,softmax_out:Matrix,grad:Matrix) -> bool:
        if softmax_out.rows != 1 and softmax_out.cols != 1:
            return False
        
        s = softmax_out.data
        jacobian = np.diagflat(s) - s @ s.T

        out.data += jacobian @ grad.data
        return True
        
    @staticmethod
    def mat_cross_entropy_add_grad(p_grad:Matrix,q_grad:Matrix,
                                   p:Matrix,q:Matrix,grad:Matrix) -> bool:
        if p.rows != q.rows or p.cols != q.cols:
            return False
        
        if p_grad is not None:
            if p_grad.rows != p.rows or p_grad.cols != p.cols:
                return False
            p_grad.data += -np.log(q.data) * grad.data

        if q_grad is not None:
            if q_grad.rows != q.rows or q_grad.cols != q.cols:
                return False
            q_grad.data += -p.data / q.data * grad.data
        
        return True

def mv_create(model: ModelContext, rows: int, 
              cols: int, flags: MVFlag) -> ModelVar:
    flags = MVFlag(flags)

    out = ModelVar(
        index=model.num_vars,
        flags=flags,
        val=Operations.mat_create(rows, cols),
        op=MVOp.CREATE,
    )
    model.num_vars += 1

    if MVFlag.REQUIRES_GRAD in flags:
        out.grad = Operations.mat_create(rows, cols)

    if MVFlag.INPUT in flags:          model.input = out
    if MVFlag.OUTPUT in flags:         model.output = out
    if MVFlag.DESIRED_OUTPUT in flags: model.desired_output = out
    if MVFlag.COST in flags:           model.cost = out

    return out

def mv_unary_impl(model:ModelContext,input_var:ModelVar,rows:int
                  ,cols:int,flags:MVFlag,op:MVOp) -> ModelVar:
    flags = MVFlag(flags)
    if MVFlag.REQUIRES_GRAD in input_var.flags:
        flags |= MVFlag.REQUIRES_GRAD

    out = mv_create(model, rows, cols, flags)
    out.op = op
    out.inputs[0] = input_var
    return out

def mv_binary_impl(model:ModelContext,a:ModelVar,b:ModelVar,
                   rows:int,cols:int,flags:MVFlag,op:MVOp) -> ModelVar:
    flags = MVFlag(flags)
    if MVFlag.REQUIRES_GRAD in a.flags or MVFlag.REQUIRES_GRAD in b.flags:
        flags |= MVFlag.REQUIRES_GRAD

    out = mv_create(model,rows,cols,flags)
    out.op = op
    out.inputs[0] = a
    out.inputs[1] = b

    return out

def mv_relu(model:ModelContext,input_var:ModelVar,
            flags:MVFlag = MVFlag.NONE) -> ModelVar:
    
    return mv_unary_impl(model,input_var,input_var.val.rows,
                         input_var.val.cols,flags,MVOp.RELU)

def mv_sigmoid(model:ModelContext,input_var:ModelVar,
            flags:MVFlag = MVFlag.NONE) -> ModelVar:
    
    return mv_unary_impl(model,input_var,input_var.val.rows,
                         input_var.val.cols,flags,MVOp.SIGMOID)

def mv_tanh(model:ModelContext,input_var:ModelVar,
            flags:MVFlag = MVFlag.NONE) -> ModelVar:
    
    return mv_unary_impl(model,input_var,input_var.val.rows,
                         input_var.val.cols,flags,MVOp.TANH)

def mv_softmax(model:ModelContext,input_var:ModelVar,
               flags:MVFlag = MVFlag.NONE) -> ModelVar:
    
    return mv_unary_impl(model,input_var,input_var.val.rows,
                         input_var.val.cols,flags,MVOp.SOFTMAX)

def mv_add(model:ModelContext,a:ModelVar,b:ModelVar,
               flags:MVFlag = MVFlag.NONE) -> ModelVar:
    if a.val.rows != b.val.rows or a.val.cols != b.val.cols:
        return None
    
    return mv_binary_impl(model,a,b,a.val.rows,a.val.cols,flags,MVOp.ADD)

def mv_sub(model:ModelContext,a:ModelVar,b:ModelVar,
               flags:MVFlag = MVFlag.NONE) -> ModelVar:
    if a.val.rows != b.val.rows or a.val.cols != b.val.cols:
        return None
    
    return mv_binary_impl(model,a,b,a.val.rows,a.val.cols,flags,MVOp.SUB)

def mv_matmul(model:ModelContext,a:ModelVar,b:ModelVar,
               flags:MVFlag = MVFlag.NONE) -> ModelVar:
    if a.val.cols != b.val.rows:
        return None
    
    return mv_binary_impl(model,a,b,a.val.rows,b.val.cols,flags,MVOp.MATMUL)

def mv_cross_entropy(model:ModelContext,p:ModelVar,q:ModelVar,
               flags:MVFlag = MVFlag.NONE) -> ModelVar:
    if p.val.rows != q.val.rows or p.val.cols != q.val.cols:
        return None
    
    return mv_binary_impl(model,p,q,p.val.rows,p.val.cols,flags,MVOp.CROSS_ENTROPY)


def model_prog_create(model:ModelContext,out_var:ModelVar) -> ModelProgram:
    visited = set()
    order = []

    stack = [(out_var,False)]

    while stack:
        cur,expanded = stack.pop()

        if expanded:
            order.append(cur)
            continue

        if cur.index in visited:
            continue

        visited.add(cur.index)

        stack.append((cur,True))

        for i in range(mv_num_inputs(cur.op)):
            input_var = cur.inputs[i]
            if input_var.index not in visited:
                stack.append((input_var,False))
    
    return ModelProgram(vars=order,size=len(order))


def model_prog_compute(prog: ModelProgram) -> None:
    for cur in prog.vars:
        a = cur.inputs[0]
        b = cur.inputs[1]

        match cur.op:
            case MVOp.RELU:
                Operations.mat_relu(cur.val,a.val)
            case MVOp.SOFTMAX:
                Operations.mat_softmax(cur.val,a.val)
            case MVOp.SIGMOID:
                Operations.mat_sigmoid(cur.val,a.val)
            case MVOp.TANH:
                Operations.mat_tanh(cur.val,a.val)
            
            case MVOp.ADD:
                Operations.mat_add(cur.val,a.val,b.val)
            case MVOp.SUB:
                Operations.mat_sub(cur.val,a.val,b.val)
            case MVOp.MATMUL:
                Operations.mat_mul(cur.val,a.val,b.val,True,False,False)
            case MVOp.CROSS_ENTROPY:
                Operations.mat_cross_entropy(cur.val,a.val,b.val)
            
def model_prog_compute_grads(prog:ModelProgram) -> None:
    for cur in prog.vars:
        if MVFlag.REQUIRES_GRAD not in cur.flags:
            continue
        if MVFlag.PARAMETER in cur.flags:
            continue
        Operations.mat_clear(cur.grad)

    Operations.mat_fill(prog.vars[-1].grad,1.0)

    for cur in reversed(prog.vars):
        if MVFlag.REQUIRES_GRAD not in cur.flags:
            continue

        a = cur.inputs[0]
        b = cur.inputs[1]

        num_inputs = mv_num_inputs(cur.op)

        if num_inputs == 1 and MVFlag.REQUIRES_GRAD not in a.flags:
            continue

        if(num_inputs == 2
            and MVFlag.REQUIRES_GRAD not in a.flags
            and MVFlag.REQUIRES_GRAD not in b.flags
        ):
            continue

        match cur.op:
            case MVOp.RELU:
                Operations.mat_relu_add_grad(a.grad,a.val,cur.grad)
            
            case MVOp.SOFTMAX:
                Operations.mat_softmax_add_grad(a.grad,cur.val,cur.grad)

            case MVOp.SIGMOID:
                Operations.mat_sigmoid_add_grad(a.grad,cur.val,cur.grad)

            case MVOp.TANH:
                 Operations.mat_tanh_add_grad(a.grad,cur.val,cur.grad)
            
            case MVOp.ADD:
                if MVFlag.REQUIRES_GRAD in a.flags:
                    Operations.mat_add(a.grad,a.grad,cur.grad)
                if MVFlag.REQUIRES_GRAD in b.flags:
                    Operations.mat_add(b.grad,b.grad,cur.grad)

            case MVOp.SUB:
                if MVFlag.REQUIRES_GRAD in a.flags:
                    Operations.mat_add(a.grad,a.grad,cur.grad)
                if MVFlag.REQUIRES_GRAD in b.flags:
                    Operations.mat_sub(b.grad,b.grad,cur.grad)

            case MVOp.MATMUL:
                if MVFlag.REQUIRES_GRAD in a.flags:
                    Operations.mat_mul(a.grad,cur.grad,b.val,False,False,True)
                if MVFlag.REQUIRES_GRAD in b.flags:
                    Operations.mat_mul(b.grad,a.val,cur.grad,False,True,False)

            case MVOp.CROSS_ENTROPY:
                p = a
                q = b
                Operations.mat_cross_entropy_add_grad(
                    p.grad,q.grad,p.val,q.val,cur.grad
                )

def model_create() -> ModelContext:
    return ModelContext()

def model_compile(model:ModelContext) -> None:
    if model.output is not None:
        model.forward_prog = model_prog_create(model,model.output)

    if model.cost is not None:
        model.cost_prog = model_prog_create(model,model.cost)

def model_feedforward(model: ModelContext) -> None:
    model_prog_compute(model.forward_prog)

def model_train(model:ModelContext, training_desc:ModelTrainingDesc) -> None:
    train_images = training_desc.train_images
    train_labels = training_desc.train_labels
    test_images = training_desc.test_images
    test_labels = training_desc.test_labels

    num_examples = train_images.rows
    num_tests = test_images.rows
    batch_size = training_desc.batch_size

    num_batches = num_examples // batch_size

    rng = np.random.default_rng()
    training_order = np.arange(num_examples)

    for epoch in range(training_desc.epochs):
        rng.shuffle(training_order)

        for batch in range(num_batches):
            for cur in model.cost_prog.vars:
                if MVFlag.PARAMETER in cur.flags:
                    Operations.mat_clear(cur.grad)

            avg_cost = 0.0
            for i in range(batch_size):
                index = training_order[batch * batch_size + i]

                model.input.val.data[:] = train_images.data[index].reshape(-1,1)
                model.desired_output.val.data[:] = train_labels.data[index].reshape(-1,1)

                model_prog_compute(model.cost_prog)
                model_prog_compute_grads(model.cost_prog)

                avg_cost += Operations.mat_sum(model.cost.val)
            avg_cost /= batch_size

            for cur in model.cost_prog.vars:
                if MVFlag.PARAMETER not in cur.flags:
                    continue

                Operations.mat_scale(cur.grad,training_desc.learning_rate / batch_size)
                Operations.mat_sub(cur.val,cur.val,cur.grad)

            print(f"Epoch {epoch + 1:2d} / {training_desc.epochs:2d}, "
                f"Batch {batch + 1:4d} / {num_batches:4d}, "
                f"Average Cost: {avg_cost:.4f}",
                end="\r", flush=True)
        print()

        num_correct = 0
        avg_cost = 0.0
        for i in range(num_tests):
            model.input.val.data[:] = test_images.data[i].reshape(-1, 1)
            model.desired_output.val.data[:] = test_labels.data[i].reshape(-1, 1)
 
            model_prog_compute(model.cost_prog)
 
            avg_cost += Operations.mat_sum(model.cost.val)
            num_correct += (
                Operations.mat_argmax(model.output.val)
                == Operations.mat_argmax(model.desired_output.val)
            )
 
        avg_cost /= num_tests
        print(
            f"Test Completed. Accuracy: {num_correct:5d} / {num_tests:5d} "
            f"({num_correct / num_tests * 100.0:.1f}%), "
            f"Average Cost: {avg_cost:.4f}")

rng = np.random.default_rng()

def mat_load(rows:int, cols:int, filename:str) -> Matrix:
    data = np.fromfile(filename,dtype=np.float32, count=rows * cols)
    return Matrix(rows,cols,data.reshape(rows,cols))

def mat_fill_rand(matrix: Matrix, lower: float, upper:float) -> None:
    matrix.data[:] = rng.uniform(lower,upper,size=matrix.data.shape)

def create_mnist_model(model:ModelContext) -> None:
    input_var = mv_create(model,784,1,MVFlag.INPUT)

    W0 = mv_create(model,16,784,MVFlag.REQUIRES_GRAD | MVFlag.PARAMETER)
    W1 = mv_create(model,16,16,MVFlag.REQUIRES_GRAD | MVFlag.PARAMETER)
    W2 = mv_create(model,10,16,MVFlag.REQUIRES_GRAD | MVFlag.PARAMETER)

    bound0 = np.sqrt(6.0 / (784 + 16))
    bound1 = np.sqrt(6.0 / (16 + 16))
    bound2 = np.sqrt(6.0 / (16 + 10))
    mat_fill_rand(W0.val, -bound0, bound0)
    mat_fill_rand(W1.val, -bound1, bound1)
    mat_fill_rand(W2.val, -bound2, bound2)

    b0 = mv_create(model,16,1,MVFlag.REQUIRES_GRAD | MVFlag.PARAMETER)
    b1 = mv_create(model,16,1,MVFlag.REQUIRES_GRAD | MVFlag.PARAMETER)
    b2 = mv_create(model,10,1,MVFlag.REQUIRES_GRAD | MVFlag.PARAMETER)

    z0_a = mv_matmul(model,W0,input_var)
    z0_b = mv_add(model,z0_a,b0)
    a0 = mv_relu(model,z0_b)

    z1_a = mv_matmul(model,W1,a0)
    z1_b = mv_add(model,z1_a,b1)
    z1_c = mv_relu(model,z1_b)
    a1 = mv_add(model,a0,z1_c)

    z2_a = mv_matmul(model,W2,a1)
    z2_b = mv_add(model,z2_a,b2)
    output = mv_softmax(model,z2_b,MVFlag.OUTPUT)

    y = mv_create(model,10,1,MVFlag.DESIRED_OUTPUT)

    mv_cross_entropy(model,y,output,MVFlag.COST)

def labels_to_one_hot(labels:np.ndarray, num_classes:int = 10) -> Matrix:
    n = labels.shape[0]
    one_hot = np.zeros((n,num_classes), dtype=np.float32)
    one_hot[np.arange(n), labels.astype(np.int64)] = 1.0
    return Matrix(n,num_classes,one_hot)

def main() -> None:
    train_images = mat_load(60000,784,"train_images.mat")
    test_images = mat_load(10000,784,"test_images.mat")
    
    train_labels_file = mat_load(60000,1,"train_labels.mat")
    test_labels_file = mat_load(10000,1,"test_labels.mat")

    train_labels = labels_to_one_hot(train_labels_file.data[:,0])
    test_labels = labels_to_one_hot(test_labels_file.data[:,0])

    draw_mnist_digit(test_images.data[0])
    print(" ".join(f"{x:.0f}" for x in test_labels.data[0]))
    print()

    model = model_create()
    create_mnist_model(model)
    model_compile(model)

    model.input.val.data[:] = test_images.data[0].reshape(-1,1)
    model_feedforward(model)

    print("pre-training output: " + 
          " ".join(f"{x:.2f}" for x in model.output.val.data[:,0]))
    training_desc = ModelTrainingDesc(train_images=train_images,train_labels=train_labels,
                                      test_images=test_images,test_labels=test_labels,
                                      epochs=10, batch_size=50,learning_rate=0.01,)
    
    model_train(model,training_desc)

    model.input.val.data[:] = test_images.data[0].reshape(-1,1)
    model_feedforward(model)
    print("post-training output: " + " ".join(f"{x:f}" for x in model.output.val.data[:, 0]))
    print()

if __name__ == "__main__":
    main()




    

    

    


        


    
    


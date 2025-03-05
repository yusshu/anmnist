import Matrix from "@/lib/Matrix";

export interface ActivationFunction {
  (m: Matrix): Matrix;

  derivative(m: Matrix): Matrix;
}

const softmax: ActivationFunction = (m: Matrix) => {
  const result = new Matrix(m.m, m.n);
  let sum = 0;
  for (let i = 0; i < m.m; i++) {
    for (let j = 0; j < m.n; j++) {
      sum += Math.exp(m.get(i, j));
    }
  }
  for (let i = 0; i < m.m; i++) {
    for (let j = 0; j < m.n; j++) {
      result.set(i, j, Math.exp(m.get(i, j)) / sum);
    }
  }
  return result;
};

softmax.derivative = (m: Matrix) => {
  return new Matrix(m.m, m.n).add(1);
};

const relu = (m: Matrix) => {
  const result = new Matrix(m.m, m.n);
  for (let i = 0; i < m.m; i++) {
    for (let j = 0; j < m.n; j++) {
      result.set(i, j, Math.max(0, m.get(i, j)));
    }
  }
  return result;
}

relu.derivative = (m: Matrix) => {
  const result = new Matrix(m.m, m.n);
  for (let i = 0; i < m.m; i++) {
    for (let j = 0; j < m.n; j++) {
      result.set(i, j, m.get(i, j) > 0 ? 1 : 0);
    }
  }
  return result;
};

export { softmax, relu };
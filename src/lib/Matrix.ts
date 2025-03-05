export default class Matrix {
  public readonly data: Array<number>;

  constructor(
    public readonly m: number,
    public readonly n: number
  ) {
    this.data = new Array(m * n).fill(0);
  }

  get(i: number, j: number) {
    return this.data[i * this.n + j];
  }

  set(i: number, j: number, value: number) {
    this.data[i * this.n + j] = value;
  }

  /**
   * Fill the matrix with random double values between min and max
   *
   * @param min {number} The minimum value
   * @param max {number} The maximum value
   */
  fillRandom(min: number, max: number) {
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        this.set(i, j, Math.random() * (max - min) + min);
      }
    }
  }

  /**
   * Multiply this matrix with another matrix or a scalar
   * and get a new matrix
   *
   * @param multiplicand {number | Matrix} the matrix or scalar to multiply with
   */
  mul(multiplicand: number | Matrix): Matrix {
    if (typeof multiplicand === 'number') {
      const result = new Matrix(this.m, this.n);
      for (let i = 0; i < this.m; i++) {
        for (let j = 0; j < this.n; j++) {
          result.set(i, j, this.get(i, j) * multiplicand);
        }
      }
      return result;
    } else {
      if (this.n !== multiplicand.m) {
        throw new Error(`Matrix dimensions do not match (this: ${this.m}x${this.n}, that: ${multiplicand.m}x${multiplicand.n})`);
      }
      const result = new Matrix(this.m, multiplicand.n);
      for (let i = 0; i < this.m; i++) {
        for (let j = 0; j < multiplicand.n; j++) {
          let sum = 0;
          for (let k = 0; k < this.n; k++) {
            sum += this.get(i, k) * multiplicand.get(k, j);
          }
          result.set(i, j, sum);
        }
      }
      return result;
    }
  }

  /**
   * Add this matrix to another matrix or a scalar, adding
   * to a scalar will add the scalar to each element of the
   * matrix. The method will return a new matrix with the
   * result and will not modify the original matrix.
   *
   * @param addend {Matrix | number} the matrix or scalar
   * @returns {Matrix} the result of the addition
   */
  add(addend: Matrix | number): Matrix {
    if (typeof addend === 'number') {
      const result = new Matrix(this.m, this.n);
      for (let i = 0; i < this.m; i++) {
        for (let j = 0; j < this.n; j++) {
          result.set(i, j, this.get(i, j) + addend);
        }
      }
      return result;
    } else {
      if (this.m !== addend.m || this.n !== addend.n) {
        throw new Error('Matrix dimensions do not match');
      }
      const result = new Matrix(this.m, this.n);
      for (let i = 0; i < this.m; i++) {
        for (let j = 0; j < this.n; j++) {
          result.set(i, j, this.get(i, j) + addend.get(i, j));
        }
      }
      return result;
    }
  }

  /**
   * Subtract a matrix or a scalar from this matrix. Subtracting
   * a scalar will subtract the scalar from each element of the
   * matrix. The method will return a new matrix with the result
   * and will not modify the original matrix.
   *
   * @param subtrahend {Matrix | number} the matrix or scalar
   * @returns {Matrix} the result of the subtraction
   */
  sub(subtrahend: Matrix | number): Matrix {
    if (typeof subtrahend === 'number') {
      const result = new Matrix(this.m, this.n);
      for (let i = 0; i < this.m; i++) {
        for (let j = 0; j < this.n; j++) {
          result.set(i, j, this.get(i, j) - subtrahend);
        }
      }
      return result;
    } else {
      if (this.m !== subtrahend.m || this.n !== subtrahend.n) {
        throwMatrixDimensionsDoNotMatch(this, subtrahend);
      }
      const result = new Matrix(this.m, this.n);
      for (let i = 0; i < this.m; i++) {
        for (let j = 0; j < this.n; j++) {
          result.set(i, j, this.get(i, j) - subtrahend.get(i, j));
        }
      }
      return result;
    }
  }

  /**
   * Compute the Hadamard product of this matrix with another
   * matrix. The Hadamard product is the element-wise product
   * of two matrices of the same dimensions.
   *
   * @param y {Matrix} the matrix to multiply with
   * @returns {Matrix} the result of the Hadamard product
   */
  hadamard(y: Matrix): Matrix {
    if (this.m !== y.m || this.n !== y.n) {
      throwMatrixDimensionsDoNotMatch(this, y);
    }
    const result = new Matrix(this.m, this.n);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.set(i, j, this.get(i, j) * y.get(i, j));
      }
    }
    return result
  }

  /**
   * Transpose this matrix and return a new matrix with the
   * transposed values. The original matrix will not be modified.
   *
   * @returns {Matrix} the transposed matrix
   */
  transpose(): Matrix {
    const result = new Matrix(this.n, this.m);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.set(j, i, this.get(i, j));
      }
    }
    return result;
  }

  argmax() {
    let max = this.get(0, 0);
    let maxIndex = 0;
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        if (this.get(i, j) > max) {
          max = this.get(i, j);
          maxIndex = i;
        }
      }
    }
    return maxIndex;
  }

  anyNaN() {
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        if (isNaN(this.get(i, j))) {
          return true;
        }
      }
    }
    return false;
  }

  toJS() {
    let result = '[';
    for (let i = 0; i < this.data.length; i++) {
      if (i !== 0) result += ',';
      result += this.data[i];
    }
    return result + ']';
  }

  static from(arr: number[], m: number, n: number) {
    const matrix = new Matrix(m, n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        matrix.set(i, j, Number(arr[i * n + j]));
      }
    }
    return matrix;
  }

  static filling(func: (i: number, j: number) => number, m: number, n: number) {
    const matrix = new Matrix(m, n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        matrix.set(i, j, func(i, j));
      }
    }
    return matrix;
  }
}

function throwMatrixDimensionsDoNotMatch(thisOne: Matrix, that: Matrix): never {
  throw new Error(`Matrix dimensions do not match (this: ${thisOne.m}x${thisOne.n}, that: ${that.m}x${that.n})`);
}